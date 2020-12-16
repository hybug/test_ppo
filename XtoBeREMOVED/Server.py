# coding: utf-8

import os
import logging
import time
import glob
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def clean(episodic, lifelong, data_dir, postfix):
    if lifelong is not None:
        if episodic >= lifelong:
            episodic -= lifelong
            sEPID = str(episodic)
            sEPID = (8 - len(sEPID)) * "0" + sEPID
            pattern = os.path.join(data_dir, sEPID + "_*." + postfix)
            names = glob.glob(pattern)
            for name in names:
                if os.path.exists(name):
                    try:
                        os.remove(name)
                    except FileNotFoundError:
                        pass


def run(**kwargs):
    tmplimit = 512
    lifelong = None

    SCRIPT_DIR = kwargs.get("SCRIPT_DIR")
    BASE_DIR = kwargs.get("BASE_DIR")
    CKPT_DIR = kwargs.get("CKPT_DIR")
    DATA_DIR = kwargs.get("DATA_DIR")

    logging.basicConfig(
        filename=os.path.join(
            BASE_DIR, "Serverlog"),
        level="INFO")

    frames = kwargs.get("frames")
    workers = kwargs.get("workers")
    parallel = kwargs.get("worker_parallel")
    MAX_STEPS = kwargs.get("max_steps")
    SEQLEN = kwargs.get("seqlen")
    CLIP = kwargs.get("vf_clip")

    episodic = 0
    sleep = 20.0

    old_segs = None
    while episodic < 100000000:
        clean(episodic, lifelong, DATA_DIR, "seg")
        clean(episodic, 8, DATA_DIR, "log")

        segs = len(glob.glob(os.path.join(DATA_DIR, "*.seg")))
        if segs > 2.5 * tmplimit:
            _s = 1
            logging.info("Episodic %d, Segs %d, SLEEP %d seconds" % (episodic, segs, _s))
            time.sleep(_s)
            continue

        if segs < 1.25 * tmplimit:
            sleep -= 0.5
            sleep = max(sleep, 1)
        elif segs > 2 * tmplimit:
            sleep += 0.5
        elif old_segs is not None:
            if segs > old_segs:
                sleep += 0.1
            else:
                sleep -= 0.1

        old_segs = segs

        for p in range(parallel):
            sEPID = str(episodic)
            sEPID = (8 - len(sEPID)) * "0" + sEPID
            sp = str(p)
            sp = (2 - len(sp)) * "0" + sp
            sEPID += sp
            for i in range(workers):
                sWKID = str(i)
                sWKID = (4 - len(sWKID)) * "0" + sWKID
                sPREID = sEPID + "_" + sWKID

                cmd = "python3 -u %s/Worker.py " \
                      "-BASE_DIR %s " \
                      "-CKPT_DIR %s " \
                      "-DATA_DIR %s " \
                      "-max_segs %d " \
                      "-sPREID %s " \
                      "-seqlen %d " \
                      "-frames %d " \
                      "-MAX_STEPS %d " \
                      "-CLIP %.4f > %s/%s.log " \
                      "2>&1 &" % (
                          SCRIPT_DIR,
                          BASE_DIR,
                          CKPT_DIR,
                          DATA_DIR,
                          3.5 * tmplimit,
                          sPREID,
                          SEQLEN,
                          frames,
                          MAX_STEPS,
                          CLIP,
                          DATA_DIR,
                          sPREID
                      )
                os.system(cmd)

        logging.info("Episodic %d, Worker %d, SLEEP %.4f s" % (episodic, workers, sleep))
        episodic += 1
        time.sleep(sleep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-SCRIPT_DIR", type=str)
    parser.add_argument("-BASE_DIR", type=str)
    parser.add_argument("-CKPT_DIR", type=str)
    parser.add_argument("-DATA_DIR", type=str)
    parser.add_argument("-frames", type=int)
    parser.add_argument("-workers", type=int)
    parser.add_argument("-worker_parallel", type=int)
    parser.add_argument("-max_steps", type=int)
    parser.add_argument("-seqlen", type=int)
    parser.add_argument("-vf_clip", type=float)
    args = parser.parse_args()
    run(**args.__dict__)
    pass
