import logging
import sys
from collections.abc import Iterable
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from tqdm import tqdm
# noinspection PyProtectedMember
from tqdm.contrib import DummyTqdmFile as StdTqdmFile
from tqdm.std import tqdm as std_tqdm
from typing import Union, Optional, Type, List, Iterator


class _TqdmLoggingHandler(logging.StreamHandler):
    def __init__(
            self,
            tqdm_class=std_tqdm  # type: Type[std_tqdm]
    ):
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            # self.tqdm_class.write(msg)
            # this change was necessary to resolve issues with the progress bars.
            self.tqdm_class.write(msg, file=self.stream)
            # self.tqdm_class.write(msg, file=sys.__stderr__)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa pylint: disable=bare-except
            self.handleError(record)


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    __slots__ = ("file", "progress")

    def __init__(self, file, progress):
        self.file = file
        self.progress = progress

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            # self.progress.write(str(type(self.file)).strip(), file=self.file)
            # self.progress.write(str(self.file).strip(), file=self.file)
            # self.progress.write(repr(self.file).rstrip(), file=self.file)
            self.progress.write(x.strip(), file=self.file)

    def flush(self):
        getattr(self.file, "flush", lambda: None)()


DummyFileType = Union[StdTqdmFile, DummyTqdmFile]


def _is_console_logging_handler(handler):
    if not isinstance(handler, logging.StreamHandler):
        return False
    # print(type(sys.stdout), type(handler.stream))
    # print(handler.stream)
    # print(handler.stream.name)
    # print(type(handler.stream.name))
    # print(handler.stream == sys.stdout)
    if handler.stream in {sys.stdout, sys.stderr}:
        return True
    elif handler.stream.name in {'<stdout>', '<stderr>'}:
        return True
    # print("Not a console logging handler.")
    return False

    # return (isinstance(handler, logging.StreamHandler)
    #         and handler.stream in {sys.stdout, sys.stderr})


def _get_first_found_console_logging_handler(handlers):
    # noinspection PyInconsistentReturns
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler


# noinspection PyUnusedLocal
@contextmanager
def tqdm_redirect(progress):
    orig_out_err = sys.stdout, sys.stderr
    try:
        # sys.stdout = DummyTqdmFile(sys.stdout, progress)
        # sys.stderr = DummyTqdmFile(sys.stderr, progress)
        sys.stdout = StdTqdmFile(sys.stdout)
        sys.stderr = StdTqdmFile(sys.stderr)
        yield orig_out_err[0]
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


# noinspection PyUnusedLocal
@contextmanager
def new_tqdm_redirect(progress):
    if not (isinstance(sys.stdout, DummyFileType) or isinstance(sys.stderr, DummyFileType)):
        dummy_file = StdTqdmFile(sys.stdout)
        dummy_error = StdTqdmFile(sys.stderr)
        with redirect_stdout(dummy_file) as orig_out, redirect_stderr(dummy_error):
            yield orig_out
    elif isinstance(sys.stdout, DummyTqdmFile):
        yield sys.stdout.file
    elif isinstance(sys.stdout, StdTqdmFile):
        # noinspection PyProtectedMember
        yield sys.stdout._wrapped


@contextmanager
def logging_redirect_tqdm(
        loggers=None,  # type: Optional[List[logging.Logger]],
        tqdm_class=std_tqdm,  # type: Type[std_tqdm]
        dummy_file=None
):
    # type: (...) -> Iterator[None]
    """
    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional
    dummy_file  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_class)
            if dummy_file is not None:
                print("compare the process wrappers")
                print(dummy_file.progress, tqdm_handler.tqdm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None and isinstance(orig_handler, logging.StreamHandler):
                assert hasattr(orig_handler, 'stream')
                assert hasattr(tqdm_handler, 'stream')
                print("TYPE STREAM: ", type(orig_handler), orig_handler.stream)
                print(f"Redirect the logging for logger: {logger.name}")
                tqdm_handler.setFormatter(orig_handler.formatter)
                # tqdm_handler.stream = orig_handler.stream
                tqdm_handler.stream = sys.stderr
            logger.handlers = [
                                  handler for handler in logger.handlers
                                  if not _is_console_logging_handler(handler)] + [tqdm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers


# noinspection PyUnusedLocal
@contextmanager
def logging_writing_redirect(loggers=None, tqdm_class=std_tqdm, **kwargs):
    with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
        with new_tqdm_redirect(tqdm_class) as orig_stream:
            yield orig_stream


# noinspection PyIncorrectDocstring
@contextmanager
def tqdm_logging_redirect(
        *args,
        # loggers=None,  # type: Optional[List[logging.Logger]]
        # tqdm=None,  # type: Optional[Type[tqdm.tqdm]]
        **kwargs
):
    # type: (...) -> Iterator[None]
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """
    tqdm_kwargs = kwargs.copy()
    loggers = tqdm_kwargs.pop('loggers', None)
    tqdm_class = tqdm_kwargs.pop('tqdm_class', std_tqdm)
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar

# noinspection PyIncorrectDocstring
@contextmanager
def tqdm_logging_writing_redirect(
        *args,
        # loggers=None,  # type: Optional[List[logging.Logger]]
        # tqdm=None,  # type: Optional[Type[tqdm.tqdm]]
        **kwargs
):
    # type: (...) -> Iterator[None]
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """
    tqdm_kwargs = kwargs.copy()
    loggers = tqdm_kwargs.pop('loggers', None)
    tqdm_class = tqdm_kwargs.pop('tqdm_class', std_tqdm)
    with tqdm_class(*args, **tqdm_kwargs) as pbar:

        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar

def advanced_tqdm_iterator(*args, tqdm_class=tqdm, logger: Optional[logging.Logger]=None, **tqdm_kwargs):
    loggers = tqdm_kwargs.pop('loggers', None)
    if loggers is None:
        loggers = [logging.root]
    if logger is not None:
        loggers.append(logger)
    postfix_formatter = tqdm_kwargs.pop('postfix_formatter', "{}")
    postfix_iterator = tqdm_kwargs.pop("iterator_postfix", False)
    pre_iteration_hook = tqdm_kwargs.pop("pre_iteration_hook", None)
    post_iteration_hook = tqdm_kwargs.pop("post_iteration_hook", None)
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            if pre_iteration_hook is not None and callable(pre_iteration_hook):
                pre_iteration_hook(pbar)
            for i in pbar:
                if postfix_iterator and isinstance(i, Iterable):
                    pbar.set_postfix(iteration=postfix_formatter.format(*i))
                else:
                    pbar.set_postfix(iteration=postfix_formatter.format(i))
                yield i

            if post_iteration_hook is not None and callable(post_iteration_hook):
                post_iteration_hook(pbar)
            if logger is not None:
                try:
                    logger.info(str(pbar))
                    pbar.colour = 'red'
                except:
                    logger.exception("Could not log the progress bar final state")





