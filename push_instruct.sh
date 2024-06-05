#!/bin/bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github_gmon2
git push origin master