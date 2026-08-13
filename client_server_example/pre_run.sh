#! /bin/bash

mkfifo client_to_server
mkfifo server_to_client

# open a subshell in the background and finish
(
read line < server_to_client
# the server said it is ready, now start the client
echo "starting the client"
) &

