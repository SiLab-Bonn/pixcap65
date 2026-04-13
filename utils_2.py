import tables as tb


def walk_to_node(parent: tb.Node, path: str, create=False):
    result = parent
    for element in path.split('/'):
        if element in result:
            result = result[element]
        elif create:
                result = parent._v_file.create_group(where=result, name=element)
        else:
            raise AssertionError("The requested path does not exist and creating the path is disabled.")
    return result
