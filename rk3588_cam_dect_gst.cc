#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
// #include <opencv2/opencv.hpp>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i) {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

/*-------------------------------------------
                  Globals
-------------------------------------------*/
rknn_context   ctx;
int            img_width = 0;
int            img_height = 0;
int            img_channel = 0;
int            width = 0;
int            height = 0;
int            channel = 3;
rknn_input_output_num io_num;
rknn_tensor_attr input_attrs[1];
rknn_tensor_attr output_attrs[3];
std::vector<float> out_scales;
std::vector<int32_t> out_zps;
float           box_conf_threshold = 0.35;
float           nms_threshold = 0.5;

GstElement *appsrc;

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
static GstFlowReturn on_new_sample(GstAppSink *sink, gpointer user_data)
{
    // printf("IN\n");
    cv::Mat frame;
    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_ERROR;

    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *structure = gst_caps_get_structure(caps, 0);
    gint width, height;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);
    cv::Mat temp(height, width, CV_8UC2, map.data); // Note: CV_8UC2 for YUY2
    cv::cvtColor(temp, frame, cv::COLOR_YUV2BGR_YUY2);
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    // printf("Received frame: %dx%dx%d\n", width, height, channel);
    // printf("need size: %dx%dx%d\n\n",  img_width, img_height, img_channel);
    // Resize if necessary
    void* resize_buf = nullptr;
    if (frame.cols != img_width || frame.rows != img_height) {
        resize_buf = malloc(height * width * channel);
        rga_buffer_t src = wrapbuffer_virtualaddr((void*)frame.data, frame.cols, frame.rows, RK_FORMAT_RGB_888);
        rga_buffer_t dst = wrapbuffer_virtualaddr(resize_buf, img_width, img_height, RK_FORMAT_RGB_888);
        im_rect src_rect = {0, 0, frame.cols, frame.rows};
        im_rect dst_rect = {0, 0, img_width, img_height};
        imresize(src, dst);
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img_width * img_height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = resize_buf ? resize_buf : frame.data;

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 0;
    }

    rknn_run(ctx, NULL);
    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    float scale_w = (float)img_width / frame.cols;
    float scale_h = (float)img_height / frame.rows;

    detect_result_group_t detect_result_group;

    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, img_height, img_width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(frame, det_result->name, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255));
        printf("%s", det_result->name);
        printf("box: (%d, %d) (%d, %d)\n", x1, y1, x2, y2);
    }

    if (resize_buf) {
        free(resize_buf);
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    // Convert OpenCV frame to GstBuffer
    GstBuffer *out_buffer = gst_buffer_new_allocate(NULL, frame.total() * frame.elemSize(), NULL);
    gst_buffer_fill(out_buffer, 0, frame.data, frame.total() * frame.elemSize());

    GstFlowReturn ret;
    g_signal_emit_by_name(appsrc, "push-buffer", out_buffer, &ret);
    gst_buffer_unref(out_buffer);

    // printf("OUT\n");
    return GST_FLOW_OK;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s <rknn model>\n", argv[0]);
        return -1;
    }

    char* model_name = argv[1];

    int            model_data_size = 0;
    unsigned char* model_data      = load_model(model_name, &model_data_size);
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret != 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));

        out_zps.push_back(output_attrs[i].zp);
        out_scales.push_back(output_attrs[i].scale);
    }

    img_width  = input_attrs[0].dims[2];
    img_height = input_attrs[0].dims[1];
    img_channel = input_attrs[0].dims[3];

    // GStreamer initialization and main loop
    gst_init(&argc, &argv);
    
    GstElement *pipeline = gst_parse_launch("v4l2src device=/dev/video21 ! video/x-raw,format=YUY2 ! appsink name=sink", NULL);
    GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");

    g_object_set(sink, "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), NULL);

    // Display pipeline
    GstElement *display_pipeline = gst_parse_launch("appsrc name=src ! videoconvert ! waylandsink name=wsink", NULL);
    appsrc = gst_bin_get_by_name(GST_BIN(display_pipeline), "src");

    // Set the caps for appsrc
    g_object_set(appsrc, "caps",
        gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, 1280,
            "height", G_TYPE_INT, 720,
            NULL), NULL);

    // Get the waylandsink element and set it to fullscreen
    GstElement *wayland_sink = gst_bin_get_by_name(GST_BIN(display_pipeline), "wsink");
    g_object_set(wayland_sink, "fullscreen", TRUE, NULL); 

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    gst_element_set_state(display_pipeline, GST_STATE_PLAYING);
    g_main_loop_run(g_main_loop_new(NULL, FALSE));

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_element_set_state(display_pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    gst_object_unref(display_pipeline);

    if (model_data) {
        free(model_data);
    }
    rknn_destroy(ctx);

    return 0;
}

