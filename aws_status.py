#!/usr/bin/env python3

#%%
import os
import subprocess
import json
from collections import defaultdict

machine_names = {
    'i-0050a301e2eb97154': '4xV100 16gb',
}

def get_awsdata():
    return json.loads(subprocess.run(["aws", "ec2", "describe-instances"], capture_output=True).stdout)

def parse_awsdata(data):
    data = data.get('Reservations')
    if not data:
        return
    infos = []
    for reservation in data:
        instances = reservation.get('Instances')
        if not instances:
            continue
        for instance in instances:
            info = defaultdict(lambda: None)
            if 'State' in instance:
                info['StateCode'] = instance['State'].get('Code')
                info['StateName'] = instance['State'].get('Name')
            info['State'] = instance.get('State')
            info['InstanceId'] = instance.get('InstanceId')
            if info['InstanceId'] not in machine_names:
                break
            info['Name'] = machine_names.get(info['InstanceId'], "!!Unnamed Instance!!")
            info['InstanceType'] = instance.get('InstanceType')
            info['PublicDnsName'] = instance.get('PublicDnsName')
            infos.append(info)
    return infos

def print_summary():
    data = parse_awsdata(get_awsdata())
    for entry in data:
        print(f"Name: {entry['Name']} (ID: {entry['InstanceId']})")
        print(f"{'':>10}State: {entry['StateName']} (Code: {entry['StateCode']})")
        print(f"{'':>10}Type: {entry['InstanceType']}")
        print(f"{'':>10}URL: {entry['PublicDnsName']}")
        print()
# %%
if __name__ == "__main__":
    print_summary()
