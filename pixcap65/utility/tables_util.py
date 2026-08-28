"""
Utility functions for :py:mod:`pytables` to access protected properties.
"""
import tables as tb
try:
    # noinspection PyCompatibility
    from collections.abc import ItemsView
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from typing import ItemsView
from typing import Any


def get_groups(parent: tb.Group) -> ItemsView[str, tb.Group]:
    """
    get_groups

    @author: Dominik Fischer
    last update: 2026-08-26

    Get the groups in the given parent group.
    :param parent: tables Group for which children groups should be returned.
    :return: mapping of child group names to the group!
    """
    # noinspection PyProtectedMember
    return parent._v_groups.items()


def get_leaves(parent: tb.Group) -> ItemsView[str, tb.Leaf]:
    """
    get_leaves

    @author: Dominik Fischer
    last update: 2026-08-26

    Get the Leaves in the given parent group.
    :param parent: tables Group for which children leaves should be returned.
    :return: mapping of child leave names to the leave!
    """
    # noinspection PyProtectedMember
    return parent._v_leaves.items()


def get_children(parent: tb.Group) -> ItemsView[str, tb.Node]:
    """
    get_children

    @author: Dominik Fischer
    last update: 2026-08-26

    Get the children in the given parent group.
    :param parent: parent from which children should be returned.
    :return: children as a view of group names and objects.
    """
    # noinspection PyProtectedMember
    return parent._v_children.items()


def get_children_bare(parent: tb.Group):
    """
    get_children_bare

    @author: Dominik Fischer
    last update: 2026-08-26

    Get the children in the given parent group.
    The children will be returned as mapping of childrens name to object.
    :param parent: parent from which children should be returned.
    :return: dictionary of all the children nodes.
    :rtype: dict[str, tb.Node]
    """
    # noinspection PyProtectedMember
    return parent._v_children


def copy_node(node: tb.Node, **kwargs) -> tb.Node:
    """
    copy_node

    @author: Dominik Fischer
    last update: 2026-08-26

    Copy this node and return the new node.
    :param node: node to be copied.
    :param kwargs: for the additional possible keyword arguments see the pytables documentation on
    :py:meth:`tables.Node._f_copy`.
    :return: newly created node.
    """
    # noinspection PyProtectedMember
    return node._f_copy(**kwargs)


def list_attributes(leave: tb.Leaf) -> list[str]:
    """
    list all the attributes associated with the given leave.

    @author: Dominik Fischer
    last update: 2026-08-26

    :param leave: leave for which the attributes should be returned.
    :return: list of attributes (names of these)
    """
    # noinspection PyProtectedMember
    return leave.attrs._f_list()


def group_get_file(group: tb.Group) -> tb.File:
    """
    group_get_file

    @author: Dominik Fischer
    last update: 2026-08-26

    Fetches the hosting File instance.

    :param group: group for which the hosting file is requested.
    :return: hosting file instance
    :rtype: :py:class:`pytables.File`
    """
    # noinspection PyProtectedMember
    return group._v_file


def set_group_attribute(group: tb.Group, attr: str, value: Any) -> None:
    """
    set_group_attribute

    @author: Dominik Fischer
    last update: 2026-08-26

    Set a PyTables attribute for this node.
    If the node already has a large number of attributes, a PerformanceWarning is issued.
    :param group: hdf file group/node for which an attribute should be set.
    :param attr: name of the attribute to set.
    :param value: value of the attribute to set.
    """
    # noinspection PyProtectedMember
    group._f_setattr(attr, value)


def get_group_attribute(group: tb.Group, attr: str) -> Any:
    """
    get_group_attribute

    @author: Dominik Fischer
    last update: 2026-08-26

    Get a PyTables attribute from this node.

    If the named attribute does not exist, an AttributeError is raised.

    :param group: hdf file group/node for which an attribute should be returned.
    :param attr: name of the attribute to get.
    :return: value of the attribute.
    """
    # noinspection PyProtectedMember
    return group._f_getattr(attr)


def get_group_attributes(group: tb.Group) -> Any:
    """
    get_group_attributes

    @author: Dominik Fischer
    last update: 2026-08-26

    AttributeSet instance associated to the Node.

    :param group: group for which attributes should be returned.
    :return: AttributeSet instance
    """
    # noinspection PyProtectedMember
    return group._v_attrs


def list_group_attributes(group: tb.Group) -> list[str]:
    """
    list group_attributes

    @author: Dominik Fischer
    last update: 2026-08-26

    List all the attributes associated with the given node (Will be fetched from the associated AttributeSet)
    :param group: group for which attributes should be returned.
    :return: list of all the attribute names.
    """
    return get_group_attributes(group)._f_list()


def get_parent_group(group: tb.Group) -> tb.Group:
    """
    get_parent_group

    @author: Dominik Fischer
    last update: 2026-08-26

    Return the parent Group instance.

    :param group: node for which the parent group should be returned.
    :return: parent group in the files hierarchy.
    """
    # noinspection PyProtectedMember
    return group._v_parent


def get_node_pathname(node: tb.Node) -> str:
    """
    get_node_pathname

    @author: Dominik Fischer
    last update: 2026-08-26

    Gets the full path in the hdf file of the given node.
    :param node: node for which the pathname should be returned.
    :return: pathname of the node.
    """
    # noinspection PyProtectedMember
    return node._v_pathname


def hdf_get_or_create_path(file: tb.File, *args, **kwargs):
    """
    hdf_get_or_create_path

    @author: Dominik Fischer
    last update: 2026-08-26

    Get the group/node associated with the specified path or create this group.
    :param file: file in which to look for the group.
    :param args: positional arguments to pass to the file.
    :param kwargs: keyword arguments to pass to the file.
    :return: fetched or newly created group.
    """
    # noinspection PyProtectedMember
    return file._get_or_create_path(*args, **kwargs)


def rename_node(node: tb.Node, new_name: str):
    """
    rename_node

    @author: Dominik Fischer
    last update: 2026-08-26

    Rename this node in place.

    Changes the name of a node to newname (a string).
    If a node with the same newname already exists and overwrite is true, recursively remove it before renaming.

    :param node: node to be renamed.
    :param new_name: new name for the node.
    """
    if isinstance(node, tb.Leaf):
        node.rename(new_name)
    else:
        # noinspection PyProtectedMember
        node._f_rename(new_name)

def back_node(node: tb.Group, group: tb.Group, rename_target=None):
    """
    back_node

    @author: Dominik Fischer
    last update: 2026-08-26

    create a backup of this node, e.g. to override or delete it.

    :param node: the node which should be backed up (including it's children)
    :param group: parent group/group in which the backup is necessary.
    :param rename_target: ??
    """
    for key, value in get_children(node):
        new_name = key
        while new_name in group:
            print("old name: ", new_name)
            new_name = "{old}_backing".format(old=new_name)
            print("new name: ", new_name)

        rename_node(value if rename_target is None else rename_target, new_name)
