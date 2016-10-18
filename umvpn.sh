#!/bin/bash

lpass show --name umich.edu --password | sudo openconnect --authgroup=Tunnel-UM-Only -u lijiax --passwd-on-stdin umvpn.umnet.umich.edu
