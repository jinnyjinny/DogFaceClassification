def sigol(input, label, model, detector, predictor):
    data = []
    ratio = []
    ##### Breed Predict #####
    breed_list = os.listdir(label)
    num_classes = len(breed_list)
    # print("{} breeds".format(num_classes))

    n_total_images = 0
    for breed in breed_list:
        n_total_images += len(os.listdir(label + "/{}".format(breed)))
    # print("{} images".format(n_total_images))

    label_maps = {}
    label_maps_rev = {}
    for i, v in enumerate(breed_list):
        label_maps.update({v: i})
        label_maps_rev.update({i: v})

        ##### Load Image #####
    img = cv2.imread(input)
    breed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ##### Predict image #####
    breed_img = imread(input)
    filename = os.path.splitext(os.path.basename(input))[0]
    breed_img = preprocess_input(breed_img)
    probs = model.predict(np.expand_dims(breed_img, axis=0))

    for idx in probs.argsort()[0][::-1][:1]:
        breed = label_maps_rev[idx].split("-")[-1]

    ##### Load Model #####
    detector = dlib.cnn_face_detection_model_v1(detector)
    predictor = dlib.shape_predictor(predictor)

    ##### Detect Face #####
    try:
        dets = detector(img, upsample_num_times=2)
    except RuntimeWarning as e:
        pass
    img_result = img.copy()

    for i, d in enumerate(dets):
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255, 0, 0), lineType=cv2.LINE_AA)

    ##### Detect Landmarks #####
    for i, d in enumerate(dets):
        shape = predictor(img, d.rect)  # detect in range of d.rect
        shape = face_utils.shape_to_np(shape)
        shape = shape.reshape(-1)  # x0 y0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        ##### Landmarks Ratio #####
        x3 = int(shape[6])
        y3 = int(shape[7])
        x5 = int(shape[10])
        y5 = int(shape[11])
        x2 = int(shape[4])
        y2 = int(shape[5])
        area = abs((x5 - x3) * (y2 - y3) - (y5 - y3) * (x2 - x3))
        AB = ((x5 - x2) ** 2 + (y5 - y2) ** 2) ** 0.5
        ratio = (area / AB) * 0.01
        ratio = round(ratio, 3)

    ##### RGB Pixel #####
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    k = 5
    clt = KMeans(n_clusters=k)
    clt.fit(img)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    dictionary = {}
    for (percent, color) in zip(hist, clt.cluster_centers_):
        startX = 0
        endX = startX + (percent * 300)
        block = endX - startX
        dictionary[block] = color.astype("uint8")
        startX = endX

    rgb_max = max(dictionary.keys())
    rgb_max = dictionary[rgb_max]
    r = rgb_max[0]
    g = rgb_max[1]
    b = rgb_max[2]

    R = round(r / (r + g + b), 3)
    G = round(g / (r + g + b), 3)
    B = round(b / (r + g + b), 3)

    data = [filename, ratio, R, G, B, breed]
    # print(data)
    return data