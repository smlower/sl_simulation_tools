#!/usr/bin/env python3
"""Provide a formatted report on slurm resource availability and usage.

Author: Oleksandr Moskalenko <om@rc.ufl.edu>, 2017-2021
Contributors: Benjamin Kimock <kimockb@ufl.edu>
"""

import sys
import logging
from loguru import logger
import argparse
import getpass
import grp
import subprocess
import os
import pwd
import shlex
import textwrap
import shutil
import io
from collections import defaultdict, namedtuple
from tokenize import tokenize

# CONFING
PARTITION_BLACKLIST = ["total"]
DEFAULT_PARTITION = "hpg-default"
DEFAULT_PARTITIONS = {"hpg-default,hpg2-compute": "hpg-default",
                      "hpg-default,hpg2-compute,hpg-milan": "hpg-default"}

DUPLICATE_PARTITIONS = {'hpg-ai': 'gpu'}
SLURM_BIN = "/opt/slurm/bin"
USE_PATH = False
H_LINE = "-" * 70
__VERSION = "21.8.30"


# CODE
def check_python_version():
    """Make sure the minimum python version requirement is met"""
    if sys.version_info < (3, 4, 0):
        print("You need python 3.4 or later to run this script.")
        sys.exit(1)


def check_slurm_binaries(args):
    """
    Verify that the binaries we need are in the PATH or our standard location
    before calling them:
    scontrol, squeue, sinfo, sacctmgr
    """
    global USE_PATH
    log = args.log
    exe_list = ["scontrol", "squeue", "sinfo", "sacctmgr"]
    res = None
    for exe in exe_list:
        res = shutil.which(exe)
        if not res:
            res = shutil.which(exe, path=SLURM_BIN)
            if not res:
                log.error(
                    """SLURM executable '{}' is not available. Check your $PATH (or log out
                          and log back in) and try again. Contact RC support if the issue
                          persists.""".format(
                        exe
                    )
                )
                sys.exit(1)
            else:
                USE_PATH = True


def _get_primary_group():
    """Return primary group name for the specified user or the caller"""
    username = getpass.getuser()
    user_record = pwd.getpwnam(username)
    group_record = grp.getgrgid(user_record.pw_gid)
    group_name = group_record.gr_name
    return group_name


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options] [group] | -g <group>",
        description="Provide a slurm resource usage and availability report for a group",
    )
    parser.add_argument("-l", "--logfile", help="Log file")
    parser.add_argument(dest="mygroup", nargs="?", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "-g",
        "--group",
        required=False,
        type=str,
        help="A different group to provide a report for",
    )
    parser.add_argument(
        "-p",
        "--partitions",
        action="store_true",
        default=False,
        help="Show resource usage for each partition",
    )
    parser.add_argument(
        "-a",
        "--allocation",
        action="store_true",
        default=False,
        help="Only show the investment allocations",
    )
    parser.add_argument(
        "-u",
        "--users",
        action="store_true",
        default=False,
        help="Show resource usage for each user",
    )
    parser.add_argument(
        "-s",
        "--sort",
        dest="sort_by",
        default="cpu",
        choices=["cpu", "mem"],
        help="When showing users sort by cpu or mem (Default: cpu)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s Version: {version}".format(version=__VERSION),
    )
    args = parser.parse_args()
    if not args.group:
        if args.mygroup:
            if args.verbose:
                print("Selecting '{}' group".format(args.mygroup))
            args.group = args.mygroup
    return args


def setup_logger(args):
    """Set up logging to a file and to stdout if verbose output is selected"""
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    global log
    log = logging.getLogger(__name__)
    if args.logfile:
        logfile = args.logfile
        file_log = logging.FileHandler(logfile)
        file_log.setFormatter(formatter)
        log.addHandler(file_log)
    console_log = logging.StreamHandler(stream=sys.stdout)
    console_log.setFormatter(formatter)
    if args.debug:
        console_log.setLevel(logging.DEBUG)
        log.setLevel("DEBUG")
    else:
        console_log.setLevel(logging.INFO)
        log.setLevel("INFO")
    log.addHandler(console_log)
    return log


def error_exit(msg):
    print("")
    log.error("{}\n".format(msg))
    sys.exit(1)


def run_command(args, cmd):
    """Run a command with subprocess and return stdout."""
    if USE_PATH:
        cmd[0] = os.path.join(SLURM_BIN, cmd[0])
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ) as proc:
        stdout = proc.stdout.read().strip()
        stderr = proc.stderr.read()
    if stderr:
        log.error("Error: stderr returned: {}".format(stderr))
    return stdout


def get_allocation_data(args, group):
    """
    Return qos allocation information for output to screen
     ['ufhpc', '31-00:00:00', 'cpu=5000,gres/gpu=20,mem=18000000M']
     ['mcintyre', '31-00:00:00', 'cpu=186,mem=669600M']
    '['christou', '00:00:00', 'cpu=0,mem=0']'
    Data could be empty if a group has no allocation - ['']

    """
    log = args.log
    data = get_qos_allocation_data(args, group)
    if args.debug:
        log.debug("Alloc data:\n'{}'".format(data))
    if len(data) > 1:
        name = data[0]
        if data[1] == "00:00:00":
            days = 0
            # If no time limit group has no allocation
            name, days, cpu, mem_mb, gpu = (group, 0, 0, 0, 0)
            return {"name": name, "time": days, "cpu": cpu, "mem": mem_mb, "gpu": gpu}
        else:
            days = data[1].split("-")[0]
        raw_tres = data[2].split(",")
        cpu = raw_tres[0].split("=")[1]
        if len(raw_tres) == 2:
            mem_idx = 1
            gpu_idx = None
        else:
            mem_idx = 2
            gpu_idx = 1
        raw_mem = raw_tres[mem_idx].split("=")[1]
        mem_mb = calculate_memory(args, raw_mem)
        if gpu_idx:
            gpu = raw_tres[gpu_idx].split("=")[1]
        else:
            gpu = 0
    else:
        name, days, cpu, mem_mb, gpu = (group, 0, 0, 0, 0)
    return {"name": name, "time": days, "cpu": cpu, "mem": mem_mb, "gpu": gpu}


def get_qos_allocation_data(args, qos):
    """Return allocation data for a requested QOS. E.g.
    '['Name|MaxWall|GrpTRES']
    '['mcintyre', '31-00:00:00', 'cpu=186,mem=669600M']'
    '['ufhpc', '31-00:00:00', 'cpu=5000,gres/gpu=20,mem=18000000M']'
    Note that the order of gres items changes if GPU gres is present.
    """
    log = args.log
    cmd_line = 'sacctmgr -P -n show qos {} format="Name,MaxWall,GrpTres"'.format(qos)
    cmd = shlex.split(cmd_line)
    res_stdout = run_command(args, cmd)
    for result in res_stdout.split("\n"):
        if args.debug:
            log.debug("Raw Allocation: {}".format(result))
        raw_data = result.strip().split("|")
    return raw_data


def get_cluster_data(args):
    """Get cluster load data for all usable partitions."""
    cluster_data = {}
    # cpu_used, cpu_total, mem_used, mem_total
    cluster_data["total"] = [0, 0, 0, 0]
    cmd_line = "sinfo --hide -h -N -O nodelist:15,partition:15,cpusstate:15,memory:10,allocmem:10"
    cmd = shlex.split(cmd_line)
    stdout = run_command(args, cmd)
    raw_data = stdout.strip().split("\n")
    # NODELIST       PARTITION      CPUS(A/I/O/T)  MEMORY    ALLOCMEM
    # c21a-s36       hpg2-compute*  18/14/0/32     128000    89700
    data = []
    for line in raw_data:
        data.append([x.strip() for x in line.split()])
    for line in data:
        # ['c35a-s18', 'hpg2-compute*', '32/0/0/32', '128000', '64000']
        # ['c0702a-s8', 'hpg-default*', '128/0/0/128', '1028000', '441024'],
        partition = line[1]
        if partition.endswith("*"):
            partition = partition[:-1]
        if partition in PARTITION_BLACKLIST:
            continue
        if partition not in cluster_data:
            cluster_data[partition] = [0, 0, 0, 0]
        # allocated/idle/other/total
        cpus = line[2].split("/")
        cpu_alloc = int(cpus[0]) + int(cpus[2])
        cpu_total = int(cpus[3])
        mem_alloc = int(int(line[4]) / 1024.0)
        mem_total = int(int(line[3]) / 1024.0)
        cluster_data[partition][0] = cluster_data[partition][0] + cpu_alloc
        cluster_data[partition][1] = cluster_data[partition][1] + cpu_total
        cluster_data[partition][2] = cluster_data[partition][2] + mem_alloc
        cluster_data[partition][3] = cluster_data[partition][3] + mem_total
        if partition not in DUPLICATE_PARTITIONS:
            cluster_data["total"][0] = cluster_data["total"][0] + cpu_alloc
            cluster_data["total"][1] = cluster_data["total"][1] + cpu_total
            cluster_data["total"][2] = cluster_data["total"][2] + mem_alloc
            cluster_data["total"][3] = cluster_data["total"][3] + mem_total
    return cluster_data


def calculate_memory(args, raw, nodes=0, cpus=0):
    """Calculate the memory request depending on units used. Convert everything
    to MB for simplicity."""
    # Only use MB for memory calculations
    # No units == M
    shift_mem_unit = False
    try:
        memory = int(raw)
    except ValueError:
        shift_mem_unit = True
    iostring = io.BytesIO(raw.encode("utf-8"))
    tokens = tokenize(iostring.readline)
    # TokenInfo(type=59 (ENCODING), string='utf-8', start=(0, 0), end=(0, 0), line='')
    # TokenInfo(type=2 (NUMBER), string='15', start=(1, 0), end=(1, 2), line='15G')
    # TokenInfo(type=1 (NAME), string='G', start=(1, 2), end=(1, 3), line='15G')
    # TokenInfo(type=0 (ENDMARKER), string='', start=(2, 0), end=(2, 0), line='')
    # > 1T
    # 2018-02-26 21:52:58,663 [DEBUG] name, val: '2', '1.17'
    # 2018-02-26 21:52:58,663 [DEBUG] name, val: '1', 'T'
    memory_units = ["K", "M", "G", "T", "KB", "MB", "GB", "Mn", "Mc", "Gn", "Gc", "N"]
    for tkname, tkval, _, _, _ in tokens:
        # if args.debug:
        #     log.debug("name, val: '{}', '{}'".format(tkname, tkval))
        if int(tkname) == 2:
            if shift_mem_unit:
                value = float(tkval)
            else:
                value = int(tkval)
        elif int(tkname) == 1:
            if tkval not in memory_units:  # ['Mn', 'Mc', 'Gn', 'Gc']:
                unit = tkval.upper()
            else:
                unit = tkval
    if unit not in memory_units:
        raise ValueError("'{}' is not a valid SLURM memory unit".format(unit))
    if unit.startswith("T"):  # in ['TB', 'T']:
        mem = int(1024 ** 2 * value)
    elif unit.startswith("G"):  # in ['GB', 'G', 'Gn', 'Gc']:
        mem = int(value) * 1024
    elif unit.startswith("M"):  # in ['M', 'MB', 'Mc', 'Mn']:
        mem = int(value)
    elif unit.startswith("K"):  # in ['K', 'KB']:
        mem = int(value / 1024.0)
    elif unit.startswith("N"):
        mem = int(value)
    else:
        raise ValueError("Use 'M', 'MB', 'G', 'GB', 'T', 'TB', or 'N' as a memory unit")
    if unit.endswith("n"):
        multiplier = 1
    elif unit.endswith("c"):
        multiplier = cpus
    else:
        multiplier = 1
    memory = mem * multiplier
    return memory


def get_queue_data(args, group, partitions):
    """Return reformatted data from squeue output for a group - total and split
    for each partition in the 'partitions' list.
    Memory usage is in MB.
    """
    inv_output, burst_output, total_output, user_output = {}, {}, {}, {}
    log = args.log
    burst = "{}-b".format(group)
    # cmd_line = "squeue -r -h --state=R,PD --account={} -O state,partition:.40,qos:.30,maxnodes,
    #            "maxcpus,minmemory,username,gres".format(group)
    # cmd_line = f"{script_dir}/squeue -r -h --state=R,PD --account={group} -O
    # state:.10,partition:.30,qos:.30,maxnodes:.5,maxcpus:.6,minmemory:.15,username:.30,gres:.15"
    # Change from -O to --format, see APPS-39 in Jira and bz40319
    # cmd_line = f"squeue -r -h --state=R,PD --account={group} -O
    # state:.10,partition:.30,qos:.30,maxnodes:.5,maxcpus:.6,minmemory:.15,username:.30,gres:.15"
    # cmd_line = f"squeue -r -h --state=R,PD --account={group} --format
    # %.10t,%.30P,%.30q,%.5D,%.6C,%.15m,%.30u #,gres:.15"
    # squeue -r -h --state=R,PD --account=ufhpc -O
    # JobID:.10,StateCompact:.2,Partition:.15,QOS:.25,MinCPUs:.6,MinMemory:.8,UserName:.25,
    # tres-alloc:.72,NodeList:.96
    # 7542315 R            gpu                    ufhpc   128  1999Gn                 ericeric
    # cpu=2048,mem=31984G,node=16,billing=2048,gres/gpu=128
    # c0901a-s[29,35],c0903a-s[11,17,23,29,35],c1003a-s23,c1009a-s[17,29,35],c1010a-s[11,17,23,29,35]
    cmd_line = (f"squeue -r -h --state=R,PD --account={group} -O State:.14,Partition:.45,QOS:"
                f".45,MaxNodes:.6,MinCPUs:.6,MinMemory:.8,UserName:.30,tres-alloc:.96"
                )
    if args.debug:
        log.debug("CMD: {}".format(cmd_line))
    cmd = shlex.split(cmd_line)
    stdout = run_command(args, cmd)
    total_output = {}
    inv_output = {}
    burst_output = {}
    user_output = {}
    gpu_output = {}
    gpu_output["investment"] = {"running": 0, "pending": 0}
    gpu_output["burst"] = {"running": 0, "pending": 0}
    gpu_output["total"] = {"running": 0, "pending": 0}
    total_output["total"] = {}
    total_output["total"]["running"] = [0, 0]
    total_output["total"]["pending"] = [0, 0]
    total_output["investment"] = {}
    total_output["investment"]["running"] = [0, 0]
    total_output["investment"]["pending"] = [0, 0]
    total_output["burst"] = {}
    total_output["burst"]["running"] = [0, 0]
    total_output["burst"]["pending"] = [0, 0]
    for partition in partitions:
        inv_output[partition] = {}
        inv_output[partition]["running"] = [0, 0]
        inv_output[partition]["pending"] = [0, 0]
        burst_output[partition] = {}
        burst_output[partition]["running"] = [0, 0]
        burst_output[partition]["pending"] = [0, 0]
    if stdout:
        job_data = stdout.strip().split("\n")
    else:
        if args.verbose:
            log.info("No jobs found for any QOSes in the '{}' account".format(group))
        # Return empty dictionaries if no jobs found
        return ({}, {}, {}, {}, {})
    for line in job_data:
        job = line.split()
        # ['R', 'gpu', 'ufhpc', '16', '128', '1999Gn', 'ericeric',
        # 'cpu=2048,mem=31984G,node=16,billing=2048,gres/gpu=128']
        if args.debug:
            logger.debug("Raw job data: '{}'".format(job))
        state = job[0].lower()
        partition = job[1]
        if partition in DEFAULT_PARTITIONS:
            partition = DEFAULT_PARTITIONS[partition]
        if ',' in partition:
            plist = partition.split(',')
            if DEFAULT_PARTITION in plist:
                partition = DEFAULT_PARTITION
        if partition in PARTITION_BLACKLIST:
            continue
        try:
            if len(job) == 8:
                raw_tres = job[7]
                user = job[6]
            else:
                user = job[5]
                raw_tres = job[-1]
            tres = dict(x.split('=') for x in raw_tres.split(','))
            if args.debug:
                logger.debug("TRES: {}".format(tres))
            qos = job[2]
            # nodes = int(job[3])
            # cpu = int(job[4]) * nodes
            # raw_memory = job[5]
            cpu = int(tres['cpu'])
            nodes = int(tres['node'])
            raw_memory = tres['mem']
        except ValueError as e:
            logger.debug(e)
            sys.exit("ERROR: Invalid data has been returned by squeue.")
            continue
        try:
            if 'gres/gpu' in tres:
                gpu = int(tres['gres/gpu'])
            else:
                gpu = 0
        except IndexError:
            gpu = "N/A"
        if gpu == "(null)" or gpu == "N/A":
            gpu = 0
        if args.debug:
            logger.debug("Processed: '[{},{},{},{},{},{},{}]'".format(qos, user, partition,
                                                                      nodes, cpu, raw_memory, gpu)
                         )
        # sys.exit("DEBUG")
        mem = calculate_memory(args, raw_memory, nodes, cpu)
        # if args.debug:
        #     log.debug("Calculated job memory: {}".format(mem))
        total_output["total"][state][0] += cpu
        total_output["total"][state][1] += mem
        if user not in user_output:
            user_output[user] = defaultdict(lambda: defaultdict(dict))
            user_output[user]["investment"]["running"] = [0, 0]
            user_output[user]["investment"]["pending"] = [0, 0]
            user_output[user]["burst"]["running"] = [0, 0]
            user_output[user]["burst"]["pending"] = [0, 0]
            # Invert qos and state order for GPUs for simplicity
            user_output[user]["gpu"]["running"] = [0, 0]
            user_output[user]["gpu"]["pending"] = [0, 0]
        if qos == group:
            qos_type = "investment"
            inv_output[partition][state][0] += cpu
            inv_output[partition][state][1] += mem
            user_output[user]["gpu"][state][0] += gpu
        elif qos == burst or "nolimit":
            qos_type = "burst"
            burst_output[partition][state][0] += cpu
            burst_output[partition][state][1] += mem
            user_output[user]["gpu"][state][1] += gpu
        else:
            if args.verbose:
                log.info("Skipping unknown qos: {}".format(line))
            continue
        # if args.debug:
        #     log.debug("Job: '{},{},{},{}'".format(qos, user, cpu, mem))
        total_output[qos_type][state][0] += cpu
        total_output[qos_type][state][1] += mem
        user_output[user][qos_type][state][0] += cpu
        user_output[user][qos_type][state][1] += mem
        gpu_output[qos_type][state] += gpu
        gpu_output["total"][state] += gpu
    return inv_output, burst_output, total_output, user_output, gpu_output


def print_use_data(data):
    """
    Format the columns of group or user usage uniformly
    The largest group name is 18 characters (early 2021).
    """
    print("{:>24}: {:>5} {:>8.0f} {:>5} {:>8.0f} {:>5} {:>8.0f}".format(*data))


def print_use(args, total_use, qos_use, account, qos_name):
    """Generate the print view of the usage data for a partition"""
    total_run_cpu = int(total_use[qos_name]["running"][0])
    total_pend_cpu = int(total_use[qos_name]["pending"][0])
    total_run_mem = int(total_use[qos_name]["running"][1]) / 1024.0
    total_pend_mem = int(total_use[qos_name]["pending"][1]) / 1024.0
    total_use_cpu = total_run_cpu + total_pend_cpu
    total_use_mem = total_run_mem + total_pend_mem
    if qos_name == "burst":
        qos_type = "-b"
        qos_name = "{}*".format(qos_name)
    else:
        qos_type = ""
    slurm_qos = "{}{}".format(account, qos_type)
    qos_header = " {} ({})".format(qos_name.title(), slurm_qos)
    total_data = (
        qos_header,
        total_run_cpu,
        total_run_mem,
        total_pend_cpu,
        total_pend_mem,
        total_use_cpu,
        total_use_mem,
    )
    number_of_used_partitions = 0
    if args.partitions:
        print("{}".format(qos_header))
        for i in qos_use:
            partition_cpu_use = int(qos_use[i]["running"][0]) + int(
                qos_use[i]["pending"][0]
            )
            if partition_cpu_use == 0:
                if not args.verbose:
                    continue
            number_of_used_partitions += 1
            partition = i
            run_cpu = qos_use[i]["running"][0]
            run_mem = int(qos_use[i]["running"][1]) / 1024.0
            pend_cpu = qos_use[i]["pending"][0]
            pend_mem = int(qos_use[i]["pending"][1]) / 1024.0
            total_cpu = run_cpu + pend_cpu
            total_mem = run_mem + pend_mem
            if total_cpu > 0:
                partition_data = (
                    partition,
                    run_cpu,
                    run_mem,
                    pend_cpu,
                    pend_mem,
                    total_cpu,
                    total_mem,
                )
                print_use_data(partition_data)
        if number_of_used_partitions > 1:
            total_data = (
                "Total",
                total_run_cpu,
                total_run_mem,
                total_pend_cpu,
                total_pend_mem,
                total_use_cpu,
                total_use_mem,
            )
            print_use_data(total_data)
    else:
        print_use_data(total_data)


def print_user_use(args, user_use, account, print_inv, print_burst):
    """Print invividual user resource usage."""
    total_cpu = print_inv + print_burst
    if total_cpu > 0:
        print("Individual usage:")
    print_qos = {"investment": print_inv, "burst": print_burst}
    qos_account = {"investment": account, "burst": "{}-b".format(account)}
    for qos in ["investment", "burst"]:
        usage = []
        sorted_usage = []
        for uname in user_use:
            run_t = tuple(user_use[uname][qos]["running"])
            pend_t = tuple(user_use[uname][qos]["pending"])
            run_cpu, run_mem = run_t
            run_mem = run_mem / 1024.0
            pend_cpu, pend_mem = pend_t
            pend_mem = pend_mem / 1024.0
            tot_cpu = run_cpu + pend_cpu
            tot_mem = run_mem + pend_mem
            usage.append(
                [uname, run_cpu, run_mem, pend_cpu, pend_mem, tot_cpu, tot_mem]
            )
        if args.sort_by == "cpu":
            sorted_usage = sorted(usage, key=lambda x: (x[5], x[6]), reverse=True)
        else:
            sorted_usage = sorted(usage, key=lambda x: (x[6], x[5]), reverse=True)
        if print_qos[qos] > 0:
            print(" {} ({})".format(qos.title(), qos_account[qos]))
            for i in sorted_usage:
                if i[5] != 0:
                    print_use_data(i)


def print_gpu_usage(args, account, gpu_use, user_use):
    """Print gpu usage - both group and individual"""
    all_user_gpu_data = {}
    print(H_LINE)
    total_run, total_pend, total_sum = 0, 0, 0
    for qos in ["investment", "burst"]:
        # if qos == "burst":
        #    qos_name = "{}*".format(qos.title())
        # else:
        #    qos_name = qos.title()
        qos_use_total = gpu_use[qos]["running"] + gpu_use[qos]["pending"]
        total_run += gpu_use[qos]["running"]
        total_pend += gpu_use[qos]["pending"]
        total_sum += qos_use_total
    if total_sum > 0:
        print(f"Account GPU usage: {total_run:>14}{total_pend:>16}{total_sum:>14}")
    if args.users:
        user_gpu_ind_data = namedtuple(
            "user_gpu_ind_data",
            [
                "inv_run",
                "inv_pend",
                "burst_run",
                "burst_pend",
                "total_run",
                "total_pend",
                "grand_total",
            ],
        )
        for user in user_use:
            user_gpu_data = user_use[user]["gpu"]
            user_gpu_inv_run = user_gpu_data["running"][0]
            user_gpu_inv_pend = user_gpu_data["pending"][0]
            user_gpu_burst_run = user_gpu_data["running"][1]
            user_gpu_burst_pend = user_gpu_data["pending"][1]
            user_gpu_total_run = user_gpu_inv_run + user_gpu_burst_run
            user_gpu_total_pend = user_gpu_inv_pend + user_gpu_burst_pend
            user_gpu_total = user_gpu_total_run + user_gpu_total_pend
            all_user_gpu_data[user] = user_gpu_ind_data(
                user_gpu_inv_run,
                user_gpu_inv_pend,
                user_gpu_burst_run,
                user_gpu_burst_pend,
                user_gpu_total_run,
                user_gpu_total_pend,
                user_gpu_total,
            )
        print(H_LINE)
        print("Individual GPU Usage:")
        # print_ind_usage = lambda a, b, c, d: print("{:>22} : {:>5} {:>14} {:>14}".format(a, b, c,
        # d))
        qos = "investment"
        qos_use_total = gpu_use[qos]["running"] + gpu_use[qos]["pending"]
        if total_sum > 0:
            for user in all_user_gpu_data:
                data = all_user_gpu_data[user]
                total = data.inv_run + data.inv_pend
                run = data.inv_run
                pend = data.inv_pend
                if total > 0:
                    print(f"{user:>19} : {run:>11} {pend:>15} {total:>13}")


def print_allocation(args, alloc):
    """Print allocation to stdout."""
    try:
        alloc_time_hrs = int(alloc["time"]) * 24
    except ValueError:
        alloc_time_hrs = "0"
    account_name = alloc["name"]
    allocation_view = textwrap.dedent(
        """
----------------------------------------------------------------------
Allocation summary:    Time Limit             Hardware Resources
   Investment QOS           Hours          CPU     MEM(GB)     GPU
----------------------------------------------------------------------
{:>17} {:>15} {:>12} {:>11} {:>7}
----------------------------------------------------------------------""".format(
            account_name,
            alloc_time_hrs,
            alloc["cpu"],
            int(int(alloc["mem"]) / 1024),
            alloc["gpu"],
        )
    )
    print(allocation_view)


def print_cluster_utilization(args, cluster_data):
    """
    Print the cluster utilization statistics for all partitions and total numbers.
    Format per partition: cpu_alloc, cpu_total, mem_alloc, mem_total
     """
    # print_header = lambda a, b, c: print("{:>15} {:>10} {:>32}".format(a, b, c))
    # print_usage  = lambda a, b, c, d, e, f, g: print("{:>13} : {:>6} ({:>3.0f}%) {:>7} {:>12}
    # ({:>3.0f}%) {:>12}".format(a, b, c, d, e, f, g))
    print(H_LINE)
    if args.partitions:
        partition_header = "Partition  :"
        partition_list = sorted(cluster_data.keys())
    else:
        partition_header = ""
        partition_list = ["total"]
    print("HiPerGator Utilization")
    print(
        "{:>15} {:>24} {:>29}".format(
            partition_header, "CPUs: Used (%) /  Total", "MEM(GB): Used (%) /  Total"
        )
    )
    print(H_LINE)
    default_per_width = 2
    for partition in partition_list:
        # allocated/idle/other/total
        cpu_used = cluster_data[partition][0]
        cpu_total = cluster_data[partition][1]
        cpu_per = int((cpu_used / cpu_total) * 100)
        cpu_per_width = len(str(abs(cpu_per)))
        if cpu_per_width < default_per_width:
            cpu_per_width = default_per_width
        mem_used = cluster_data[partition][2]
        mem_total = cluster_data[partition][3]
        mem_per = int((mem_used / mem_total) * 100)
        mem_per_width = len(str(abs(mem_per)))
        if mem_per_width < default_per_width:
            mem_per_width = default_per_width
        if cpu_total > 0:
            if partition == "total":
                partition = "Total"
            print(f"{partition:>13} : {cpu_used:>8} ({cpu_per:>3}%) / {cpu_total:>6} {mem_used:>13}"
                  f"({mem_per:>3}%) / {mem_total:>6}")


def show_output(
    args, alloc_data, cluster_data, total_use, inv_use, burst_use, user_use, gpu_use
):
    """Print formatted output."""
    account_name = alloc_data["name"]
    # Allocation
    print_allocation(args, alloc_data)
    if args.allocation:
        sys.exit(0)
    # Usage
    if not total_use:
        print("\nNo running jobs found.\n")
    else:
        print(
            "{} {:>22} {:>15} {:>12}".format(
                "CPU/MEM Usage:", "Running", "Pending", "Total"
            )
        )
        print(
            "{:>31} {:>8} {:>5} {:>8} {:>5} {:>8}".format(
                "CPU", "MEM(GB)", "CPU", "MEM(GB)", "CPU", "MEM(GB)"
            )
        )
        print(H_LINE)
        # Investment QOS
        inv_use_exists = int(total_use["investment"]["pending"][0]) + (
            total_use["investment"]["running"][0]
        )
        if inv_use_exists:
            print_use(args, total_use, inv_use, account_name, "investment")
        burst_use_exists = int(total_use["burst"]["pending"][0]) + (
            total_use["burst"]["running"][0]
        )
        if burst_use_exists:
            print_use(args, total_use, burst_use, account_name, "burst")
        if args.users:
            print(H_LINE)
            print_user_use(
                args, user_use, account_name, inv_use_exists, burst_use_exists
            )
        gpu_use_exists = gpu_use["total"]["running"] + gpu_use["total"]["pending"]
        if gpu_use_exists > 0:
            print_gpu_usage(args, account_name, gpu_use, user_use)
    # Cluster utilization
    print_cluster_utilization(args, cluster_data)
    print(H_LINE)
    print("* Burst QOS uses idle cores at low priority with a 4-day time limit")
    dupe_msg = "* Duplicate partition(s): "
    for i in DUPLICATE_PARTITIONS:
        dupe_msg += (f"{i} / {DUPLICATE_PARTITIONS[i]}")
        if len(DUPLICATE_PARTITIONS) > 1:
            dupe_msg += ', '
    print(dupe_msg)
    print("")
    if (len(sys.argv)) < 3:
        print("Run 'slurmInfo -h' to see all available options\n")


# MAIN
def main():
    check_python_version()
    args = parse_args()
    if args.debug:
        args.verbose = True
    log = args.log = setup_logger(args)
    check_slurm_binaries(args)
    group = args.group
    if not group:
        group = _get_primary_group()
    try:
        grp.getgrnam(group)
    except KeyError:
        log.error(
            "Group '{}' does not exist. Check the input and try again.".format(group)
        )
        sys.exit(1)
    alloc_data = get_allocation_data(args, group)
    if args.verbose:
        log.info("Querying SLURM cluster and partition usage stats .....")
    cluster_data = get_cluster_data(args)
    partitions = [x for x in cluster_data.keys() if x not in PARTITION_BLACKLIST]
    if args.debug:
        log.debug("Partitions: {}".format(partitions))
    partitions.extend(DUPLICATE_PARTITIONS.keys())
    if args.debug:
        log.debug("Partitions: {}".format(partitions))
        log.debug("Cluster Data:\n{}".format(cluster_data))
    if args.verbose:
        log.info(
            "Querying SLURM queue for resource used by running and pending jobs ....."
        )
    inv_use, burst_use, total_use, user_use, gpu_use = get_queue_data(
        args, group, partitions
    )
    show_output(
        args, alloc_data, cluster_data, total_use, inv_use, burst_use, user_use, gpu_use
    )


if __name__ == "__main__":
    main()
