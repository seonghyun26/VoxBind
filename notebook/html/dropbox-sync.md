# Dropbox Sync with rclone

Sync experiment results and docs from NCLOUD servers to the lab Dropbox (SPML) using rclone.

## Installation

rclone is installed to userspace (no sudo required):

```bash
# Find latest version
curl -sL https://github.com/rclone/rclone/releases/latest -o /dev/null -w '%{url_effective}\n'

# Download and install (replace version as needed)
curl -sL https://github.com/rclone/rclone/releases/download/v1.73.3/rclone-v1.73.3-linux-amd64.zip -o /tmp/rclone.zip
mkdir -p ~/.local/bin
unzip -o /tmp/rclone.zip -d /tmp/rclone-install
cp /tmp/rclone-install/rclone-*/rclone ~/.local/bin/
chmod +x ~/.local/bin/rclone
rm -rf /tmp/rclone.zip /tmp/rclone-install

# Verify
rclone version
```

Requires `~/.local/bin` in `$PATH` (already the case on our servers).

## Authorization

Since the servers are headless, OAuth is done in two steps:

### 1. Configure the remote on the server

```bash
rclone config
```

- `n` — new remote
- Name: `dropbox`
- Storage type: `dropbox`
- Leave client_id and client_secret blank
- Advanced config: `n`
- Auto config (web browser): `n` — this is a headless server

It will ask you to run `rclone authorize "dropbox"` on a machine with a browser.

### 2. Authorize from your laptop

On your local machine (Mac/Windows/Linux with a browser):

```bash
# Install rclone locally if needed (Mac: brew install rclone)
rclone authorize "dropbox"
```

A browser window opens. Log into Dropbox and authorize rclone. Copy the token printed in the terminal and paste it back into the server prompt.

Finish the config:

- Keep this remote: `y`
- Quit config: `q`

## Connecting to the SPML shared folder

By default, rclone only shows your personal Dropbox namespace. The SPML team folder lives under a different root namespace.

### Find the root namespace ID

```bash
TOKEN=$(python3 -c "
import json
t = open('$HOME/.config/rclone/rclone.conf').read().split('token = ')[1].split('\n')[0]
print(json.loads(t)['access_token'])
")

curl -s -X POST https://api.dropboxapi.com/2/users/get_current_account \
  --header "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool | grep -A3 root_info
```

Look for `root_namespace_id` in the output.

### Set the root namespace

```bash
rclone config update dropbox root_namespace_id <ID>
```

For our SPML team account, the root namespace ID is `12221840097`.

### Verify access

```bash
# List team root (note the leading /)
rclone lsd dropbox:/
# Should show: SPML, nayoung

# List SPML contents
rclone lsd dropbox:/SPML

# Your personal folder is at
rclone lsd dropbox:/nayoung
```

The leading `/` is required to access the team root. Without it, you see your personal namespace only.

## Syncing experiments

```bash
# Upload experiments to Dropbox (local -> Dropbox)
rclone sync experiments/ "dropbox:/nayoung/MatterFlow/experiments/" --progress

# Download from Dropbox to local (Dropbox -> local)
rclone sync "dropbox:/nayoung/MatterFlow/experiments/" experiments/ --progress
```
`rclone sync` makes the destination match the source. Use `rclone copy` instead if you don't want to delete files at the destination that are missing from the source.

## Syncing docs

```bash
# Upload experiments to Dropbox (local -> Dropbox)
rclone sync docs/ "dropbox:/nayoung/MatterFlow/docs/" --progress

# Download from Dropbox to local (Dropbox -> local)
rclone sync "dropbox:/nayoung/MatterFlow/docs/" docs/ --progress
```

### Dry run

Preview what would be transferred without actually doing it:

```bash
rclone sync experiments/ "dropbox:/SPML/nayoung/MatterFlow/experiments/" --dry-run
```

## Config location

rclone config is stored at `~/.config/rclone/rclone.conf`. This file contains your Dropbox OAuth token — do not commit it to git.
