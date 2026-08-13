#! /bin/bash

memory_overcommit_file=/proc/sys/vm/overcommit_memory

sudo_user=$(sudo bash -c 'echo $SUDO_USER')
: ${sudo_user:?"Error: You have no sudo permissions! Please try again with sudo."}

echo "Setting the overcommit policy to always..."

sudo bash -c "echo 1 > $memory_overcommit_file"

