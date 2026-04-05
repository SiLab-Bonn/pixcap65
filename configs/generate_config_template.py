import argparse
import os
from argparse import ArgumentParser
from typing import Iterable

import numpy as np
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

yml = ruamel.yaml.YAML()

smu_character = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z']
smu_numbering = np.arange(101)


def path_type(string):
    if os.path.exists(string):
        return string
    else:
        raise argparse.ArgumentTypeError("The file %s does not exist!" % string)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template', '-t', type=path_type, default='Keithley_2602a_template.yaml',
                        help='Path to the template file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to write the output to.')
    parser.add_argument('--smu_numbering', '-n', action='store_true', default=101,
                        help='Whether, to use numbering scheme for the SMU channels.')
    args = parser.parse_args()

    if args.smu_numbering:
        smu_modifier = smu_numbering
    else:
        smu_modifier = smu_character

    if args.output is None:
        if args.template.endswith('_template.yaml'):
            args.output = args.template.replace('_template.yaml', '.yaml')
        else:
            args.output = args.template.replace('.yaml', '_modified.yaml')

    with open(args.template, 'r') as f:
        config = yml.load(f)

    # extract the final comments
    final_key = list(config.keys())[-1]
    print(final_key)
    final_key_content = config[final_key]
    print(final_key_content.ca)
    if final_key_content.ca.comment is None:
        print(type(final_key_content.ca.items.values()))
        print(list(final_key_content.ca.items.values())[0])
        final_comments_temp = list(final_key_content.ca.items.values())[0]
        final_comments = []
        if isinstance(final_comments_temp, Iterable):
            for comment in final_comments_temp:
                if comment is not None:
                    final_comments.append(comment)

    else:
        final_comments = final_key_content.ca.comment
    print(final_comments)

    first_top_level_declarations = config['top_level']
    second_top_level_declarations = config.pop('top_back', CommentedMap())
    template_config = config.pop('template_config', CommentedMap())
    channel_declarations = config.pop('channel', CommentedMap())

    # now adjust the configurations
    channel_configurations = {}
    if channel_declarations and 'channels' in template_config:
        print("Handle the channels")
        assert 'channels' in template_config and len(smu_modifier) >= template_config['channels']
        assert 'replace_pattern' in template_config
        replacePattern = template_config['replace_pattern']
        searchPattern = f"{replacePattern}X"
        for i in range(template_config['channels']):
            current_channel_configuration = {}
            for key, value in channel_declarations.items():
                current_channel_configuration[key] = value.replace(searchPattern, f"{replacePattern}{smu_modifier[i]}")

            channel_configurations[f"channel {i + 1}"] = current_channel_configuration

    print(channel_configurations)

    for key, value in first_top_level_declarations.items():
        config.insert(0, key, value)

    del config['top_level']

    for key, value in channel_configurations.items():
        config[key] = value

    for key, value in second_top_level_declarations.items():
        config[key] = value

    config.yaml_end_comment_extend(final_comments, clear=False)

    with open(args.output, 'w') as f:
        yml.dump(config, f)
