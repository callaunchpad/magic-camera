#!/bin/bash
echo "Turning on aws instance"
while true; do
	echo "cycle"
	aws ec2 start-instances --instance-ids i-0050a301e2eb97154
	sleep 3
done
