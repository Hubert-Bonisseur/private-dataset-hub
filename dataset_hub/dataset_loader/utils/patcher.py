from dataset_hub.dataset_loader.utils.streaming_download_manager import (
    xbasename,
    xdirname,
    xet_parse,
    xgetsize,
    xglob,
    xgzip_open,
    xisdir,
    xisfile,
    xjoin,
    xlistdir,
    xopen,
    xpandas_read_csv,
    xpandas_read_excel,
    xPath,
    xrelpath,
    xsio_loadmat,
    xsplit,
    xsplitext,
    xwalk,
    xxml_dom_minidom_parse,
)
import datasets


def patch_with_gcloud_auth():
    datasets.download.streaming_download_manager.xbasename = xbasename
    datasets.download.streaming_download_manager.xdirname = xdirname
    datasets.download.streaming_download_manager.xet_parse = xet_parse
    datasets.download.streaming_download_manager.xgetsize = xgetsize
    datasets.download.streaming_download_manager.xglob = xglob
    datasets.download.streaming_download_manager.xgzip_open = xgzip_open
    datasets.download.streaming_download_manager.xisdir = xisdir
    datasets.download.streaming_download_manager.xisfile = xisfile
    datasets.download.streaming_download_manager.xlistdir = xlistdir
    datasets.download.streaming_download_manager.xjoin = xjoin
    datasets.download.streaming_download_manager.xopen = xopen
    datasets.download.streaming_download_manager.xpandas_read_csv = xpandas_read_csv
    datasets.download.streaming_download_manager.xpandas_read_excel = xpandas_read_excel
    datasets.download.streaming_download_manager.xPath = xPath
    datasets.download.streaming_download_manager.xrelpath = xrelpath
    datasets.download.streaming_download_manager.xsio_loadmat = xsio_loadmat
    datasets.download.streaming_download_manager.xsplit = xsplit
    datasets.download.streaming_download_manager.xsplitext = xsplitext
    datasets.download.streaming_download_manager.xwalk = xwalk
    datasets.download.streaming_download_manager.xwalk = xwalk
    datasets.download.streaming_download_manager.xxml_dom_minidom_parse = xxml_dom_minidom_parse
