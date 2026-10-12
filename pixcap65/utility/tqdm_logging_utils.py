"""
Utility implementation to simplify operations when using :py:mod:`tqdm` progress bars.
Here are context manager and helper classes to redirect the console logging handlers and prints on the standard output
or standard error.

Additional a enhancement to the `tqdm` implementations is provided which allows a postfix modification of the displayed
progress bar and could log the final state of the progress bar.
"""
import logging
import sys

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable, Iterator
except ImportError:
    # python 2.7 and < python 3.3
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable
    from typing import Iterator
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from tqdm import tqdm
# noinspection PyProtectedMember
from tqdm.contrib import DummyTqdmFile as StdTqdmFile
from tqdm.std import tqdm as std_tqdm
from typing import Union, Optional, Type, List


class _TqdmLoggingHandler(logging.StreamHandler):
    """
    Alternative Implementation of StreamHandler to log to console but not breaking the :py:mod:`tqdm` progress bars.
    """
    def __init__(
            self,
            tqdm_class=std_tqdm  # type: Type[std_tqdm]
    ):
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record):
        """
        emit the log record by writting to a tqdm objects writer, in order to not break progress bares when logging to
        console.
        :param record: log record to be written
        """
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
    """
    Dummy file-like that will write to tqdm.
    Make sure that output of logging will not break :py:mod:`tqdm` progress bars when logging to console.
    :ivar file: file-like object to write the output to.
    :ivar progress: tqdm iterator, to handle the redirect.
    """
    __slots__ = ("file", "progress")

    def __init__(self, file, progress):
        self.file = file
        self.progress = progress

    def write(self, x):
        """
        Write to the output file without breaking progress bars.
        :param x: string or bytes to write to the output file.
        """
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            # self.progress.write(str(type(self.file)).strip(), file=self.file)
            # self.progress.write(str(self.file).strip(), file=self.file)
            # self.progress.write(repr(self.file).rstrip(), file=self.file)
            self.progress.write(x.strip(), file=self.file)

    def flush(self):
        """
        Flush the data to the file,e.g. before closing the file handle.
        This will actually write the buffers to disk or the file stream.
        This method is totally propagated down to the actual file.
        """
        getattr(self.file, "flush", lambda: None)()


DummyFileType = Union[StdTqdmFile, DummyTqdmFile]


def _is_console_logging_handler(handler):
    """
    Utility function, determining whether the investigated logging handler is actually
    logging to the console or in general the standard output.

    @author: Dominik Fischer
    last update: 2026-08-27

    :param handler: :py:class:`logging.Handler` to be investigated.
    :type handler: :py:class:`logging.Handler`
    :return: whether the logging handler would log to standard output or standard error.
    :rtype: bool
    """
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
    """
    tqdm_redirect

    @author: Dominik Fischer
    last update: 2026-08-27

    Context manager to redirect (standard) `print`statements from the standard output/error to the
    :py:mod:`tqdm` progress bar object to not break the displayed progress bars.
    This implementation is closely oriented at the documented example for the `tqdm` framework.

    Currently, only the original error stream is yielded, but the general intention of implementing was to yield both
    the orginal output and the original error stream.

    :param progress: progress bar iterator object, unused by now.
    :return: (yields) original standard error stream.
    """
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
    """
    new_tqdm_redirect

    @author: Dominik Fischer
    last update: 2026-08-27

    Enhanced version of :py:func:`~pixcap65.utility.tqdm_utils.tqdm_redirect` which allows for nested repeated
    redirects.
    If a existing redirect is detected, only the streams which are not redirected yet will be redirected at all.

    :param progress: `tqdm` progress indicator, unused by now.
    :return: (yields) original standard error stream.
    """
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
                pass
                # print("compare the process wrappers")
                # print(dummy_file.progress, tqdm_handler.tqdm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None and isinstance(orig_handler, logging.StreamHandler):
                assert hasattr(orig_handler, 'stream')
                assert hasattr(tqdm_handler, 'stream')
                # print("TYPE STREAM: ", type(orig_handler), orig_handler.stream)
                # print(f"Redirect the logging for logger: {logger.name}")
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
    """
    logging_writing_redirect

    @author: Dominik Fischer
    last update: 2026-08-27

    Combined redirect of output by `print` statement and logging handlers which would
    log to (standard) output streams (console).
    For further information take a look at :py:func:`~pixcap65.utility.tqdm_utils.logging_redirect_tqdm` and
    :py:func:`~pixcap65.utility.tqdm_utils.tqdm_utils.new_tqdm_redirect`.

    Multiple (folded) redirects are handled appropriately.

    :param loggers: Which handlers to redirect (default: [logging.root]).
    :type loggers: list, optional
    :param tqdm_class: optional
    :return: (yields) original standard error stream.
    """
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
    """
    advanced_tqdm_iterator

    @author: Dominik Fischer
    last update: 2026-08-27

    Utility function, which provides a context manager object which handles the redirect of logging and `print`
    statements, as well as, the management of the :py:mod:`tqdm` iterator object.
    Also the iteration is handled here by the yielded context manager.
    Callbacks provide the ability to execute individual code directly before and directly after the iteration.
    A iteration depend message at the progress bar is implemented here as well.
    Additional keyword arguments are used to define its properties.
    After completing the iteration the last state of the progress bar will be logged if a logger was provided.
    This log record should provide information about the iteration frequency and the elapsed time.

    ..caution::
        currently only redirecting logger is implemented here.



    :param args: (starred argument),  positional arguments to be propagated to the `tqdm` iterator.
    :param tqdm_class: type of tqdm iterator to use here.
    :param logger: additional logger object to be added for the logging redirect.
    :type logger: logging.Logger, optional
    :keyword loggers: Which handlers to redirect (default: [logging.root]).
    :type loggers: list, optional
    :param tqdm_kwargs: further keyword arguments, e.g. to configure the additonal output formatters
    :keyword postfix_formatter: str, format string, which replacements must be available from the current iteration object.
    :keyword iterator_postfix: bool, whether the iteration variable should be considered an iterable for the format string.
    :keyword pre_iteration_hook: callable, callback to be executed before starting the first iteration. Take a
        :py:class:`tqdm.tqdm` object as its only argument.
    :keyword post_iteration_hook: callable, callback to be executed after stopping the last iteration. Take a
        :py:class:`tqdm.tqdm` object as its only argument.
    :keyword propagate_class: callable, type of tqdm iterator to propagate for the setup of the redirects.
    :return: yields a iterator over the provided iterable.
    """
    loggers = tqdm_kwargs.pop('loggers', None)
    if loggers is None:
        loggers = [logging.root]
    if logger is not None:
        loggers.append(logger)

    # prepare formatting of additonal messages
    postfix_formatter = tqdm_kwargs.pop('postfix_formatter', "{}")
    postfix_iterator = tqdm_kwargs.pop("iterator_postfix", False)
    pre_iteration_hook = tqdm_kwargs.pop("pre_iteration_hook", None)
    post_iteration_hook = tqdm_kwargs.pop("post_iteration_hook", None)
    propagate_tqdm_class = tqdm_kwargs.pop("propagate_class", std_tqdm)

    # init the progress bar.
    if type(std_tqdm) != type(tqdm_class):
        tqdm_kwargs["tqdm_class"] = propagate_tqdm_class

    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class if type(tqdm_class) == type(std_tqdm) else std_tqdm):
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
                    stats = pbar.format_dict
                    rate = stats["total"] / stats["elapsed"] if tqdm_kwargs.get("leave", True) else stats["rate"]
                    logger.info("[%s]: The elapsed time is %f s with a rate of %f Hz and %f s/it.;", tqdm_kwargs.get("desc", "DEFAULT"), stats["elapsed"], rate, 1 / rate)
                    pbar.colour = 'red'
                except:
                    logger.exception("Could not log the progress bar final state")
