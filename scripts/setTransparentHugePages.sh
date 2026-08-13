#! /bin/bash

thp_enabled_file=/sys/kernel/mm/transparent_hugepage/enabled

if [[ $# != 1 ]] ||
    { [[ "$1" != "always" && "$1" != "never" && "$1" != "madvise" ]]; }; then
    echo "Usage: $0 \"<always|never|madvise>\""
    exit -1
fi

sudo_user=$(sudo bash -c 'echo $SUDO_USER')
: ${sudo_user:?"Error: You have no sudo permissions! Please try again with sudo."}

echo "Setting THP to $1..."

sudo bash -c "echo $1 > $thp_enabled_file"

