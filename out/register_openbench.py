#!/usr/bin/env python3
"""Register Luminex engine, config, and network on the OpenBench server.
Run on the server as the openbench user with the venv python:
  cd /opt/openbench && . ./.env 2>/dev/null; set -a; . ./.env; set +a; \
  .venv/bin/python /tmp/register_openbench.py
Idempotent: updates existing rows instead of duplicating."""
import os, sys, hashlib, subprocess, shutil

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'OpenSite/settings')
import django
django.setup()
from OpenBench.models import Engine, EngineConfig, Network, Profile

SHA = '3872ff6522e26b50c44681235900c56bb6578880'
SRC = f'https://github.com/changcheng967/Luminex/archive/{SHA}.zip'
NET_URL = 'https://github.com/changcheng967/Luminex/releases/download/v6.3.0/luminex_sfq_i8.nnue'
MEDIA = '/opt/openbench/Media'

engine, _ = Engine.objects.update_or_create(
    name='Luminex',
    defaults=dict(source=SRC, sha=SHA, bench=206603))
print('Engine:', engine)

cfg, _ = EngineConfig.objects.update_or_create(
    name='Luminex',
    defaults=dict(nps=610000, source=SRC, enabled=True,
                  build_path='.', build_compilers='g++-13 g++-12 g++ clang++',
                  build_cpuflags='avx2 bmi2', build_systems='Linux'))
print('EngineConfig:', cfg, 'nps=', cfg.nps)

# network: download once, hash, place at Media/<SHA8>
local = '/tmp/luminex_sfq_i8.nnue'
if not os.path.exists(local):
    subprocess.run(['curl', '-fsSL', NET_URL, '-o', local], check=True)
h = hashlib.sha256(open(local, 'rb').read()).hexdigest()[:8].upper()
dest = os.path.join(MEDIA, h)
if not os.path.exists(dest):
    shutil.copyfile(local, dest)
    os.remove(local)
net, _ = Network.objects.update_or_create(
    engine='Luminex', name='luminex_sfq_i8.nnue',
    defaults=dict(sha256=h, author='changcheng967', default=True, was_default=True))
print('Network:', net)

# admin account per owner request: changcheng967 / approver; drop the
# bootstrap-time luminex account so exactly one admin exists
from django.contrib.auth.models import User
u, _ = User.objects.get_or_create(username='changcheng967')
u.set_password('chang123456')
u.is_staff = True
u.is_superuser = True
u.save()
p, _ = Profile.objects.get_or_create(user=u)
p.approver = True
p.enabled = True
p.save()
User.objects.filter(username='luminex').exclude(pk=u.pk).delete()
print('User changcheng967 approver=True; luminex account removed')
print('DONE')
