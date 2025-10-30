#!/usr/bin/env python3
"""
Script to automatically add vast.ai instances to SSH config.
Makes it easy to SSH into vast.ai machines and use VSCode remotely.
"""

import os
import sys
import re
import subprocess
from pathlib import Path


def get_or_create_ssh_key(key_name='id_ed25519'):
    """
    Get existing SSH key or create a new one.
    
    Args:
        key_name: Name of the SSH key file (default: id_ed25519)
    
    Returns:
        tuple: (private_key_path, public_key_path)
    """
    ssh_dir = Path.home() / '.ssh'
    ssh_dir.mkdir(exist_ok=True, mode=0o700)
    
    private_key = ssh_dir / key_name
    public_key = ssh_dir / f'{key_name}.pub'
    
    # Check if key exists
    if private_key.exists() and public_key.exists():
        print(f"✓ Found existing SSH key: {private_key}")
        return private_key, public_key
    
    # Generate new key
    print(f"\n⚠️  No SSH key found at {private_key}")
    response = input("Generate a new SSH key? (Y/n): ").strip().lower()
    if response == 'n':
        print("Cannot proceed without SSH key.")
        sys.exit(1)
    
    print("\nGenerating new SSH key...")
    
    # Use ed25519 (modern, secure, fast)
    try:
        result = subprocess.run([
            'ssh-keygen',
            '-t', 'ed25519',
            '-f', str(private_key),
            '-N', '',  # No passphrase for convenience
            '-C', f'vastai-key-{os.getlogin()}'
        ], capture_output=True, text=True, check=True)
        
        # Set proper permissions
        private_key.chmod(0o600)
        public_key.chmod(0o644)
        
        print(f"✓ Generated new SSH key: {private_key}")
        return private_key, public_key
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating SSH key: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def display_public_key(public_key_path):
    """Display the public key for the user to copy to vast.ai."""
    with open(public_key_path, 'r') as f:
        public_key = f.read().strip()
    
    print("\n" + "="*60)
    print("📋 YOUR PUBLIC SSH KEY (copy this to vast.ai):")
    print("="*60)
    print(public_key)
    print("="*60)
    print("\nTo add this key to vast.ai:")
    print("  1. Go to https://cloud.vast.ai/account/")
    print("  2. Navigate to 'SSH Keys' section")
    print("  3. Click 'Add SSH Key'")
    print("  4. Paste the key above")
    print("="*60 + "\n")
    
    input("Press Enter once you've added the key to vast.ai...")


def create_ssh_config_entry(host_name, ip, ssh_port, identity_file):
    """
    Create an SSH config entry.
    
    Args:
        host_name: Name for the host (e.g., 'vastai-gpu1')
        ip: IP address
        ssh_port: SSH port number
        identity_file: Path to private key file
    """
    config = f"\nHost {host_name}\n"
    config += f"    HostName {ip}\n"
    config += f"    User root\n"
    config += f"    Port {ssh_port}\n"
    config += f"    IdentityFile {identity_file}\n"
    
    # Add keep-alive settings
    config += "    ServerAliveInterval 60\n"
    config += "    ServerAliveCountMax 3\n"
    
    # Strict host key checking off for vast.ai (IPs change frequently)
    config += "    StrictHostKeyChecking no\n"
    config += "    UserKnownHostsFile=/dev/null\n"
    
    return config


def get_ssh_config_path():
    """Get the path to SSH config file."""
    ssh_dir = Path.home() / '.ssh'
    ssh_dir.mkdir(exist_ok=True, mode=0o700)
    
    config_path = ssh_dir / 'config'
    if not config_path.exists():
        config_path.touch(mode=0o600)
    
    return config_path


def host_exists_in_config(config_path, host_name):
    """Check if a host already exists in the SSH config."""
    if not config_path.exists():
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
        return re.search(rf'^Host\s+{re.escape(host_name)}\s*$', content, re.MULTILINE) is not None


def remove_host_from_config(config_path, host_name):
    """Remove an existing host from SSH config."""
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    skip = False
    
    for line in lines:
        if line.strip().startswith('Host '):
            # Check if this is the host to remove
            if re.match(rf'Host\s+{re.escape(host_name)}\s*$', line.strip()):
                skip = True
            else:
                skip = False
        
        if not skip:
            new_lines.append(line)
        elif line.strip().startswith('Host '):
            # Next host started, stop skipping
            skip = False
            new_lines.append(line)
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def add_to_ssh_config(config_entry, host_name):
    """Add entry to SSH config file."""
    config_path = get_ssh_config_path()
    
    # Check if host already exists
    if host_exists_in_config(config_path, host_name):
        response = input(f"Host '{host_name}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
        # Remove old entry
        remove_host_from_config(config_path, host_name)
    
    # Append new entry
    with open(config_path, 'a') as f:
        f.write(config_entry)
    
    return True


def main():
    print("=== Vast.ai SSH Config Generator ===\n")
    
    # Handle SSH keys first
    print("Step 1: Setting up SSH keys...")
    private_key, public_key = get_or_create_ssh_key()
    
    # Check if user already has key in vast.ai
    print("\n")
    already_added = input("Have you already added this SSH key to vast.ai? (y/N): ").strip().lower()
    
    if already_added != 'y':
        display_public_key(public_key)
    
    print("\nStep 2: Configuring SSH connection...")
    
    # Get host name
    default_name = "vastai"
    host_name = input(f"Enter a name for this host (default: {default_name}): ").strip()
    if not host_name:
        host_name = default_name
    
    # Get IP address
    ip = input("\nEnter the IP address: ").strip()
    if not ip:
        print("Error: IP address is required.")
        sys.exit(1)
    
    # Get SSH port
    ssh_port = input("Enter the SSH port: ").strip()
    if not ssh_port:
        print("Error: SSH port is required.")
        sys.exit(1)
    
    # Validate port is a number
    try:
        int(ssh_port)
    except ValueError:
        print("Error: SSH port must be a number.")
        sys.exit(1)
    
    print(f"\nSSH port: {ssh_port}")
    print(f"IP address: {ip}")
    print(f"Identity file: {private_key}")
    
    # Generate config entry
    config_entry = create_ssh_config_entry(host_name, ip, ssh_port, private_key)
    
    print("\n=== Generated SSH Config Entry ===")
    print(config_entry)
    print("===================================")
    
    response = input("\nAdd this to your SSH config? (Y/n): ").strip().lower()
    if response == 'n':
        print("Aborted.")
        sys.exit(0)
    
    if add_to_ssh_config(config_entry, host_name):
        print(f"\n✓ Successfully added '{host_name}' to SSH config!")
        print(f"\nYou can now connect with:")
        print(f"  ssh {host_name}")
        print(f"\nOr use in VSCode:")
        print(f"  1. Install 'Remote - SSH' extension")
        print(f"  2. Press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows/Linux)")
        print(f"  3. Type 'Remote-SSH: Connect to Host'")
        print(f"  4. Select '{host_name}'")
    else:
        print("\nFailed to add to SSH config.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)

