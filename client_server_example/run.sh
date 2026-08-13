#! /bin/bash

# start the server in the background
echo "starting the server"
server_pid=$!

# signal the client that the server is ready
echo 1 > server_to_client

wait $server_pid

