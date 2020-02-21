import io
import os
import numpy as np
import tensorflow as tf
import time

# https://github.com/tensorflow/models/tree/master/research/compression/image_encoder

#define model file
#model_name = "/dtascfs/m192500/compress_model/compression_residual_gru/residual_gru.pb"
model_name = "/projects/shart/digital_pathology/data/biliary/compression_residual_gru/residual_gru.pb"
# model_name = "H://ToyDataset//img_compression//compression_residual_gru//residual_gru.pb"

def get_output_tensor_names():
  name_list = ['GruBinarizer/SignBinarizer/Sign:0']
  for i in range(1, 16):
    name_list.append('GruBinarizer/SignBinarizer/Sign_{}:0'.format(i))
  return name_list

def batch_encoder(input_img_name_list, SAVE_FNAME, SAVE, iteration=1):
    nd_array_patches = np.ndarray([])
    row_num = 0
    if input_img_name_list is None  or model_name is None:
        print('\nParameter errors\n')
        return
    if iteration < 0 or iteration > 15:
        print('\n--iteration must be between 0 and 15 inclusive.\n')
        return

    with tf.Graph().as_default() as graph:
        # Load the inference model for encoding.
        with tf.gfile.FastGFile(model_name, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        _ = tf.import_graph_def(graph_def, name='')

        input_tensor = graph.get_tensor_by_name('Placeholder:0')
        outputs = [graph.get_tensor_by_name(name) for name in
                   get_output_tensor_names()]
        input_image = tf.placeholder(tf.string)

        with tf.Session(graph=graph) as sess:

            # change following lines if you want to input image-string
            ######################################
            for img_name_path in input_img_name_list:
                start_t = time.time()
                img_name = str(img_name_path.strip())
                print(img_name)
                with tf.gfile.FastGFile(img_name, 'rb') as input_image_fp:
                    input_image_str = input_image_fp.read()
            ######################################
                _, ext = os.path.splitext(img_name)
                if ext == '.png':
                    decoded_image = tf.image.decode_png(input_image, channels=3)
                elif ext == '.jpeg' or ext == '.jpg':
                    decoded_image = tf.image.decode_jpeg(input_image, channels=3)
                else:
                    assert False, 'Unsupported file format {}'.format(ext)
                decoded_image = tf.expand_dims(decoded_image, 0)

                img_array = sess.run(decoded_image, feed_dict={input_image: input_image_str})
                results = sess.run(outputs, feed_dict={input_tensor: img_array})

                results = results[0:iteration + 1]
                int_codes = np.asarray([x.astype(np.int8) for x in results])

                # Convert int codes to binary.
                int_codes = (int_codes + 1) // 2
                export = np.packbits(int_codes.reshape(-1))
                # codes.append(export)
                if row_num == 0:
                    nd_array_patches = export
                else:
                    nd_array_patches = np.vstack((nd_array_patches, export))
                row_num += 1

                end_t = time.time()
                print(str(start_t-end_t)+" per/patch")
                # if row_num == 2:
                #     break
        if SAVE:
            output = io.BytesIO()
            output_codes_name = os.path.join(SAVE_FNAME)
            np.savez_compressed(output, shape=nd_array_patches.shape, codes=nd_array_patches)
            with tf.gfile.FastGFile(output_codes_name, 'w') as code_file:
                code_file.write(output.getvalue())
    return nd_array_patches

def batch_arr_encoder(input_nd_arry, patch_size, save_name, SAVE, iteration=1):
    nd_array_patches = np.ndarray([])
    row_num = 0
    if input_nd_arry is None or model_name is None:
        print('\nParameter errors\n')
        return
    if iteration < 0 or iteration > 15:
        print('\n--iteration must be between 0 and 15 inclusive.\n')
        return
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with tf.Graph().as_default() as graph:
        # Load the inference model for encoding.
        with tf.gfile.FastGFile(model_name, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        _ = tf.import_graph_def(graph_def, name='')

        input_tensor = graph.get_tensor_by_name('Placeholder:0')
        outputs = [graph.get_tensor_by_name(name) for name in
                   get_output_tensor_names()]
        input_image = tf.placeholder(tf.string)

        with tf.Session(graph=graph) as sess:
            # change following lines if you want to input image-string
            ######################################
            total_cnt = input_nd_arry.shape[0]
            curr_cnt = 0
            for img_patch_array in input_nd_arry:
                # decoded_image = tf.convert_to_tensor(img_patch_array, dtype=tf.uint8)
                start_t = time.time()
                print("Processing %d / %d patches." % (curr_cnt, total_cnt))
                img = np.reshape(img_patch_array, patch_size).astype(np.uint8)
                img_array = np.expand_dims(img,0)
                curr_cnt += 1
                # decoded_image = tf.expand_dims(img, 0)
                #
                # img_array = sess.run(decoded_image, feed_dict={input_image: img})
                results = sess.run(outputs, feed_dict={input_tensor: img_array})
                # results = sess.run(outputs, feed_dict={input_tensor: img_patch_array})

                results = results[0:iteration + 1]
                int_codes = np.asarray([x.astype(np.int8) for x in results])

                # Convert int codes to binary.
                int_codes = (int_codes + 1) // 2
                export = np.packbits(int_codes.reshape(-1))
                # codes.append(export)
                if row_num == 0:
                    nd_array_patches = export
                else:
                    nd_array_patches = np.vstack((nd_array_patches, export))
                row_num += 1
                # if row_num == 2:
                #     break
                end_t = time.time()
                print(str(end_t -start_t)+"s per patch")
        if SAVE:
            output = io.BytesIO()
            np.savez_compressed(output, shape=nd_array_patches.shape, codes=nd_array_patches)
            with tf.gfile.FastGFile(save_name, 'w') as code_file:
                code_file.write(output.getvalue())
    return nd_array_patches


def main(self):
    # img_list = ["H://ToyDataset//testing_data//dog0.png", "H://ToyDataset//testing_data//dog1.png"]
    # save_dir = "H://ToyDataset//testing_data_coder"
    # img_list = ["/dtascfs/m192500/BiliaryCytology/test_data/1.jpg", "/dtascfs/m192500/BiliaryCytology/test_data/2.jpg"]
    # save_dir = "/dtascfs/m192500/BiliaryCytology/test_data/"
    # batch_encoder(img_list, save_dir, True)

    #img_list = ["H://ToyDataset//testing_data//0.png", "H://ToyDataset//testing_data//1.png"]
    #save_dir = "H://ToyDataset//testing_data_coder"
    # img_list = ["/dtascfs/m192500/BiliaryCytology/test_data/1.jpg", "/dtascfs/m192500/BiliaryCytology/test_data/2.jpg"]
    # save_dir = "/dtascfs/m192500/BiliaryCytology/test_data/"
    #batch_encoder(img_list, save_dir, True)

    test_samples = "/projects/shart/digital_pathology/data/biliary/test_samples.npy"
    test_samples_encode = "/projects/shart/digital_pathology/data/biliary/test_encode_samples.npy"
    img_arr_list = np.load(test_samples)
    patches_encoded = batch_arr_encoder(img_arr_list, test_samples_encode, True)

    print("debug")

if __name__ == '__main__':
    tf.app.run()






