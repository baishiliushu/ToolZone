from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


def distance_avg(gpu_, npu_):
    absolute_differences = np.abs(gpu_ - npu_)
    mean_absolute_difference = np.mean(absolute_differences)
    print("绝对差值的平均值：", mean_absolute_difference)


def run_compare(onnx_dump_file, npu_dump_file):
    print("-------------------{} vs {}-------------------".format(onnx_dump_file, npu_dump_file))
    gpu_src_data = np.loadtxt(onnx_dump_file)
    gpu_data_dim_one = gpu_src_data.reshape(1, -1)
    npu_src_data = np.loadtxt(npu_dump_file)
    npu_data_dim_one = npu_src_data.reshape(1, -1)
    _similar_value = cosine_similarity(gpu_data_dim_one, npu_data_dim_one)
    print(">>>>>Total _similar : {} <<<<<<".format(_similar_value ))
    distance_avg(gpu_data_dim_one, npu_data_dim_one)
    dist = np.sqrt(np.sum(np.square(gpu_data_dim_one - npu_data_dim_one)))
    
    print(">>>>>Total _dist: {};(avg:{})  <<<<<<".format(dist, dist / npu_data_dim_one.shape[1]))
    """
    reshaped_gpu = np.reshape(gpu_data_dim_one, (1, -1, 57))
    reshaped_npu = np.reshape(npu_data_dim_one, (1, -1, 57))
    box_conf_gpu = reshaped_gpu[..., 4]
    box_conf_npu = reshaped_npu[..., 4]
    _similar_value = cosine_similarity(box_conf_gpu, box_conf_npu)
    print("Box-conf-max gpu {}(id:{}), npu  {}(id:{})".format(np.max(box_conf_gpu), np.argmax(box_conf_gpu),np.max(box_conf_npu),  np.argmax(box_conf_npu)))
    print(">>>>> Box-conf _similar_value {}  <<<<<<".format(_similar_value))

    person_conf_gpu = reshaped_gpu[..., 5]
    person_conf_npu = reshaped_npu[..., 5]
    _similar_value = cosine_similarity(person_conf_gpu, person_conf_npu)
    print("Person-class-conf-max gpu {}(id:{}), npu  {}(id:{})".format(np.max(person_conf_gpu), np.argmax(person_conf_gpu),np.max(person_conf_npu),  np.argmax(person_conf_npu)))
    print(">>>>> Person-class-conf _similar_value {} <<<<<<".format(_similar_value))
    """

def main():
    np.set_printoptions(precision=8)
    np.set_printoptions(suppress=True)
    onnx_father_dir = "/home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/movenet-only-onnx/3-int16-nhwc/consult-mpose-cos0.5/onnx-rsts/"
    onnx_file = "input_crop_cv_nhwc.txt"  # "input_txt--Walking2/"
    nb_father_dir = "/home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/movenet-only-onnx/3-int16-nhwc/consult-mpose-cos0.5/npu-rsts/vknn-output/"
    nb_file = "p1_input_data.txt"
    run_compare(os.path.join(onnx_father_dir, onnx_file), os.path.join(nb_father_dir, nb_file))
    onnx_file = "output_nhwc.txt"
    nb_file = "p1_0.txt"
    run_compare(os.path.join(onnx_father_dir, onnx_file), os.path.join(nb_father_dir, nb_file))
    # nb_father_dir = "./onnx_inference/dump_txt-int8/label/"
    # run_compare(os.path.join(onnx_father_dir, onnx_file), os.path.join(nb_father_dir, nb_file))


if __name__ == '__main__':
    main()
