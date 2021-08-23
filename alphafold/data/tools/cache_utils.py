import os
import json
import pickle
from functools import lru_cache
import hashlib
import logging

DEFAULT_CACHE_DIR = '.cache'


@lru_cache(maxsize=32)
def hash_fp(fp):
    with open(fp, 'rb') as f:
        return md5_hash(f.read())


def md5_hash(s: str) -> str:
    return hashlib.md5(s).hexdigest()


def cache_key_with_hashed_paths(args, kwargs):
    """Hashable key of function name, args, and kwargs. For kwarg names ending
    in '_path', attempt to hash the file and set that as the key.
    """
    kw = dict()
    for k in kwargs:
        assert isinstance(k, str)
        if k.endswith('_path'):
            kw[k] = hash_fp(kwargs[k])
        else:
            kw[k] = kwargs[k]
    return (args, frozenset(kw.items()))


def order_dict(d: dict) -> tuple:
    return tuple(((k, d[k]) for k in sorted(d.keys())))

def looks_like_path(s: str, suffixes: tuple = ('_path', '_dir')) -> bool:
    if not isinstance(s, str):
        return False
    return any(s.endswith(suf) for suf in suffixes)

def normpath(s: str) -> str:
    return os.path.abspath(os.path.normpath(s))

def cache_key(args, kwargs):
    # convert path-like kwargs to their normal absolute paths
    kw = dict()
    for k, v in kwargs.items():
        if v is None:
            kw[k] = v
        elif looks_like_path(k):
            kw[k] = normpath(v)
        else:
            kw[k] = v

    obj = (args, order_dict(kw))

    # JSON serialize ordered (kw)args
    serialized = json.dumps(obj).encode('utf-8')
    logging.info(f"using serialized={serialized}")
    return md5_hash(serialized)
    

def cache_to_pckl(cache_dir=None, exclude_kw=None, use_pckl=True):
    """Caches function results to pickle file. Returns a decorator factory.
    Pickled function results are cached to a path `cache_dir/func.__name__/hash`
    where hash is hashed args and kwargs (except kwargs listed in `exclude_kw`).
    If the cache file exists, return the unpickled result. If it does not exist,
    or if environment variable `AF2_SKIP_PCKL_CACHE=1`, run the function and write
    its result to the cache.
    """
    if exclude_kw is None:
        exclude_kw = list()
    elif isinstance(exclude_kw, str):
        exclude_kw = [exclude_kw]
    
    if cache_dir is None:
        cache_dir = os.environ.get('AF2_CACHE_DIR', None)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # whether to use pickle or plain text cache
    cache_ext = 'pckl' if use_pckl is True else 'out'
    AF2_SKIP_PCKL_CACHE = os.environ.get('AF2_SKIP_PCKL_CACHE', 0)

    def decorator(fn):
        def wrapped(*args, **kwargs):
            kw = {k: kwargs[k] for k in kwargs if k not in exclude_kw}
            key = cache_key(args, kw)
            cache_fp = os.path.join(cache_dir, fn.__name__, f"{key}.{cache_ext}")

            logging.debug(f"using cache_fp={cache_fp} (AF2_SKIP_PCKL_CACHE={AF2_SKIP_PCKL_CACHE})")

            if os.path.exists(cache_fp) and not AF2_SKIP_PCKL_CACHE:
                logging.info(f"using cache at {cache_fp} instead of running {fn.__name__}")
                if use_pckl:
                    with open(cache_fp, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(cache_fp, 'r') as f:
                        return f.read()
            else:
                logging.info(f"no cache found at {cache_fp}")

            result = fn(*args, **kwargs)

            # write to cache file
            os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
            logging.info(f"saving results to {cache_fp}")
            if use_pckl:
                with open(cache_fp, 'wb') as f:
                    pickle.dump(result, f, protocol=4)
            else:
                with open(cache_fp, 'w') as f:
                    f.write(result)

            return result

        return wrapped

    return decorator

