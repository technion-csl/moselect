#! /bin/bash

if (( $# < 2 )); then
    echo "Usage: $0 \"node_number\" \"command_to_execute\""
    exit -1
fi

node_number="$1"
shift
command="$@"

bound_cpu_cores=$(cat /sys/devices/system/node/node${node_number}/cpulist)
echo "Binding the process to memory node: $node_number,"
echo "and its local CPU cores: $bound_cpu_cores"
taskset_command="taskset --cpu-list $bound_cpu_cores"
numactl_command="numactl --membind $node_number"
submit_command="$taskset_command $numactl_command"

$submit_command $command

