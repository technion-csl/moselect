#! /bin/bash

function setSystemFile {
    file_name=$1
    required_value=$2
    current_value=`cat $file_name`
    echo "Current value of $file_name: $current_value"

    if (( $current_value == $required_value )); then
        echo "Current value is set correctly"
    else
        echo "Current value is different than required"
        sudo bash -c "echo $required_value > $file_name"
        new_value=`cat $file_name`
        echo "New value of $file_name: $new_value"
    fi
}

paranoid_file=/proc/sys/kernel/perf_event_paranoid
required_paranoid_value=-1
setSystemFile $paranoid_file $required_paranoid_value

kptr_file=/proc/sys/kernel/kptr_restrict
required_kptr_value=0
setSystemFile $kptr_file $required_kptr_value

perf_cpu_file=/proc/sys/kernel/perf_cpu_time_max_percent
required_perf_cpu_value=0
setSystemFile $perf_cpu_file $required_perf_cpu_value
