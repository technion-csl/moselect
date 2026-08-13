#! /bin/bash

run() {
    #($run_mosalloc_script "$@" -- date) > /dev/null 2>&1
    $run_mosalloc_script "$@" -- date >> tmp.txt 2>&1
    echo $?
}

runAndCheckSuccess() {
    run_exit_value=$(run "$@")
    if (( $run_exit_value == 0 )); then
        echo "Success with arguments: $@"
    else
        echo "Should have succeeded, but failed with arguments: $@"
        exit -1
    fi
}

runAndCheckFailure() {
    run_exit_value=$(run "$@")
    if (( $run_exit_value != 0 )); then
        echo "Failed with arguments: $@"
    else
        echo "Should have failed, but succeeded with arguments: $@"
        exit -1
    fi
}

if (( $# != 1 )); then
    echo "Usage: $0 RUN_MOSALLOC_SCRIPT"
    exit -1
fi
run_mosalloc_script="$1"

base_page_size=$((4*1024))
large_page_size=$((2*1024*1024))
huge_page_size=$((1*1024*1024*1024))

runAndCheckSuccess
runAndCheckSuccess -aps 1GB -as2 $base_page_size -ae2 $((base_page_size+large_page_size))
runAndCheckSuccess -aps 2GB -as1 $base_page_size -ae1 $((base_page_size+huge_page_size))

# bad library name
runAndCheckFailure -l junk
# negative size
runAndCheckFailure -fps "-5GB"
# start > end
runAndCheckFailure -aps 1GB -as2 $base_page_size
# not aligned
runAndCheckFailure -aps 1GB -as2 $base_page_size -ae2 $((3*base_page_size))
# region exceeds pool size
runAndCheckFailure -aps 1GB -as1 $base_page_size -ae1 $((base_page_size+huge_page_size))
# 2MB and 1GB regions are overlapping
runAndCheckFailure -aps 2GB -as1 $base_page_size -ae1 $((base_page_size+huge_page_size)) -as2 $base_page_size -ae2 $((base_page_size+large_page_size))

