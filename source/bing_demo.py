from bing import *
import pandas as pd
import scipy.io as sio


def get_params_images():
    params_file  = '/home/harrysocool/Github/fast-rcnn/OP_methods/BING_Objectness/doc/bing_params.json'
    if not os.path.exists(params_file):
        print "Specified file for parameters %s does not exist." % params_file
        sys.exit(2)
    try:
        f = open(params_file, "r")
        params_str = f.read()
        f.close()
    except Exception as e:
        print "Error while reading parameters file %s. Exception: %s." % (params_file, e)
        sys.exit(2)
    try:
        params = json.loads(params_str)
    except Exception as e:
        print "Error while parsing parameters json file %s. Exception: %s." % (params_file, e)
        sys.exit(2)

    if not params.has_key("num_bbs"):
        params["num_bbs"] = 1500
    if not params.has_key("num_win_psz"):
        params["num_win_psz"] = 130
    return params

def bing_demo(image_filepath):
    params = get_params_images()
    params["image_file"] = image_filepath
    if not os.path.exists(params["image_file"]):
        print "Specified file for image %s does not exist." % params["image_file"]
        sys.exit(2)
    image = cv2.imread(params["image_file"])

    results_dir = params["results_dir"]
    if not os.path.exists(results_dir):
        print "The results directory that should contains weights and sizes indeces does not exist. Be sure to have already performed training. "
        sys.exit(2)

    if not os.path.exists(params["1st_stage_weights_fn"]):
        print "The weights for the first stage does not exist!"
        sys.exit(2)
    w_1st = np.genfromtxt(params["1st_stage_weights_fn"], delimiter=",").astype(np.float32)

    if not os.path.exists(params["sizes_indeces_fn"]):
        print "The sizes indices file does not exist!"
        sys.exit(2)
    sizes = np.genfromtxt(params["sizes_indeces_fn"], delimiter=",").astype(np.int32)

    if not os.path.exists(params["2nd_stage_weights_fn"]):
        print "The weights for the second stage does not exist!"
        sys.exit(2)
    f = open(params["2nd_stage_weights_fn"])
    w_str = f.read()
    f.close()
    w_2nd = json.loads(w_str)

    b = Bing(w_1st, sizes, w_2nd, num_bbs_per_size_1st_stage=params["num_win_psz"], num_bbs_final=params["num_bbs"])
    bbs, scores = b.predict(image)

    return bbs, scores

def generate_all_image():
    datasets_path = '/home/harrysocool/Github/fast-rcnn/DatabaseEars/'
    image_index_output_path = os.path.join(datasets_path, '../', 'ear_recognition/data_file/image_index_list.csv')
    mat_output_filename = os.path.join(datasets_path, '../','ear_recognition/data_file/all_boxes.mat')

    list1 = pd.read_csv(image_index_output_path, header=None).values.flatten().tolist()

    all_boxes = np.zeros((len(list1),), dtype=np.object)
    for i in range(436,437,1):
        bbs, _ = bing_demo(list1[i])
        bbs1 = np.asarray(bbs, dtype=np.double) - 1
        all_boxes[i] = bbs1
        print('No. {} image processed with {} boxes'.format(i, len(bbs)))

    sio.savemat(mat_output_filename, {'all_boxes': all_boxes})

if __name__ == '__main__':
    generate_all_image()

    # image_path = '/home/harrysocool/Pictures/7.jpg'
    # bbs, scores = bing_demo(image_path)
    # im = cv2.imread(image_path)
    # for i in range(5):
    #     bbox = bbs[i]
    #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    # cv2.imshow('frame', im)
    # cv2.waitKey(0)