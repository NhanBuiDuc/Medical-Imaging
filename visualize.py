from parse_config import ConfigParser
import argparse
import collections
import data_loader.data_loaders as module_data
import visualizer.isic_2019_visualizer as visualizer_module_data


def main(config):
    data_loader = config.init_obj('data_loader', module_data)
    visualizer = config.init_obj('visualizer', visualizer_module_data)
    visualizer.data_loader = data_loader

    visualizer.visualize()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--split', default="train", type=str,
                      help='splitting set (default: None)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
