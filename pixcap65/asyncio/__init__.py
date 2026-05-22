import asyncio
import re
import tables as tb
from contextlib import asynccontextmanager
from tables import Filters
from typing import Literal

lock = asyncio.Lock()
__open_files__ = {}
__events__ = {}
regex = r"The files '.*' is already opened"



# when using e.g. reading mode the same file could be read multiple times simultaneously.

async def __internal_tables_open_file(filename: str, mode: Literal["r", "w", "a", "r+"] = "r",
                                      title: str = "",
                                      root_uep: str = "/",
                                      filters: Filters | None = None,
                                      **kwargs,):
    while True:
        try:
            with tb.open_file(filename, mode, title, root_uep, filters, **kwargs) as file:
                yield file
                break

        except ValueError as e:
            async with lock:
                if re.match(regex, str(e), re.MULTILINE):
                    need_wait = True
                else:
                    need_wait = False
                    raise e
        else:
            need_wait = False

        if need_wait:
            await asyncio.sleep(10)




@asynccontextmanager
async def tables_open_file(filename: str,
                           mode: Literal["r", "w", "a", "r+"] = "r",
                           title: str = "",
                           root_uep: str = "/",
                           filters: Filters | None = None,
                           **kwargs,):
    timeout = kwargs.pop("timeout", None)
    if timeout:
        file = await asyncio.wait_for(__internal_tables_open_file(filename, mode, title, root_uep, filters, **kwargs), timeout)
        yield file
    else:
        yield await __internal_tables_open_file(filename, mode, title, root_uep, filters, **kwargs)







