import tables as tb
from typing import Any
from typing import ItemsView


def get_groups(parent: tb.Group) -> ItemsView[str, tb.Group]:
    """
    get_groups

    get the groups in the given parent group.
    :param parent: tables Group for which children groups should be returned.
    :return: mapping of child group names to the group!
    """
    # noinspection PyProtectedMember
    return parent._v_groups.items()


def get_leaves(parent: tb.Group) -> ItemsView[str, tb.Leaf]:
    """
    get_leaves

    get the Leaves in the given parent group.
    :param parent: tables Group for which children leaves should be returned.
    :return: mapping of child leave names to the leave!
    """
    # noinspection PyProtectedMember
    return parent._v_leaves.items()


def get_children(parent: tb.Group) -> ItemsView[str, tb.Node]:
    # noinspection PyProtectedMember
    return parent._v_children.items()


def get_children_bare(parent: tb.Group):
    # noinspection PyProtectedMember
    return parent._v_children


def copy_node(node: tb.Node, **kwargs) -> tb.Node:
    # noinspection PyProtectedMember
    return node._f_copy(**kwargs)


def list_attributes(leave: tb.Leaf) -> list[str]:
    # noinspection PyProtectedMember
    return leave.attrs._f_list()


def group_get_file(group: tb.Group) -> tb.File:
    # noinspection PyProtectedMember
    return group._v_file


def set_group_attribute(group: tb.Group, attr: str, value: Any) -> None:
    # noinspection PyProtectedMember
    group._f_setattr(attr, value)


def get_group_attribute(group: tb.Group, attr: str) -> Any:
    # noinspection PyProtectedMember
    return group._f_getattr(attr)


def get_group_attributes(group: tb.Group) -> Any:
    # noinspection PyProtectedMember
    return group._v_attrs

def list_group_attributes(group: tb.Group) -> list[str]:
    return get_group_attributes(group)._f_list()


def get_parent_group(group: tb.Group) -> tb.Group:
    # noinspection PyProtectedMember
    return group._v_parent


def get_node_pathname(node: tb.Node) -> str:
    # noinspection PyProtectedMember
    return node._v_pathname


def hdf_get_or_create_path(file: tb.File, *args, **kwargs):
    # noinspection PyProtectedMember
    return file._get_or_create_path(*args, **kwargs)


def rename_node(node: tb.Node, new_name: str):
    if isinstance(node, tb.Leaf):
        node.rename(new_name)
    else:
        # noinspection PyProtectedMember
        node._f_rename(new_name)

def back_node(node: tb.Group, group: tb.Group, rename_target=None):
    for key, value in get_children(node):
        new_name = key
        while new_name in group:
            print("old name: ", new_name)
            new_name = "{old}_backing".format(old=new_name)
            print("new name: ", new_name)

        rename_node(value if rename_target is None else rename_target, new_name)
