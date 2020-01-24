#an ugly solution but it works

def load_pretraining(net_dict, pretrained_dict):
    net_dict["contract1.conv1.weight"] = pretrained_dict["features.0.weight"]
    net_dict["contract1.conv1.bias"] = pretrained_dict["features.0.bias"]
    net_dict["contract1.conv2.weight"] = pretrained_dict["features.2.weight"]
    net_dict["contract1.conv2.bias"] = pretrained_dict["features.2.bias"]

    net_dict["contract2.conv1.weight"] = pretrained_dict["features.5.weight"]
    net_dict["contract2.conv1.bias"] = pretrained_dict["features.5.bias"]
    net_dict["contract2.conv2.weight"] = pretrained_dict["features.7.weight"]
    net_dict["contract2.conv2.bias"] = pretrained_dict["features.7.bias"]

    net_dict["contract3.conv1.weight"] = pretrained_dict["features.10.weight"]
    net_dict["contract3.conv1.bias"] = pretrained_dict["features.10.bias"]
    net_dict["contract3.conv2.weight"] = pretrained_dict["features.12.weight"]
    net_dict["contract3.conv2.bias"] = pretrained_dict["features.12.bias"]
    net_dict["contract3.conv3.weight"] = pretrained_dict["features.14.weight"]
    net_dict["contract3.conv3.bias"] = pretrained_dict["features.14.bias"]

    net_dict["contract4.conv1.weight"] = pretrained_dict["features.17.weight"]
    net_dict["contract4.conv1.bias"] = pretrained_dict["features.17.bias"]
    net_dict["contract4.conv2.weight"] = pretrained_dict["features.19.weight"]
    net_dict["contract4.conv2.bias"] = pretrained_dict["features.19.bias"]
    net_dict["contract4.conv3.weight"] = pretrained_dict["features.21.weight"]
    net_dict["contract4.conv3.bias"] = pretrained_dict["features.21.bias"]

    net_dict["contract5.conv1.weight"] = pretrained_dict["features.24.weight"]
    net_dict["contract5.conv1.bias"] = pretrained_dict["features.24.bias"]
    net_dict["contract5.conv2.weight"] = pretrained_dict["features.26.weight"]
    net_dict["contract5.conv2.bias"] = pretrained_dict["features.26.bias"]
    net_dict["contract5.conv3.weight"] = pretrained_dict["features.28.weight"]
    net_dict["contract5.conv3.bias"] = pretrained_dict["features.28.bias"]

    return net_dict
