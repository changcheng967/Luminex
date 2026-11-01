import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'OpenSite.settings')
django.setup()
from OpenBench.models import Engine, EngineConfig, Test, Result

NEWSHA = 'bd3d209feebe0c21f5d02309eb120b8a95ca0bb5'
SRC = f'https://github.com/changcheng967/Luminex/archive/{NEWSHA}.zip'
Engine.objects.update_or_create(name='Luminex', defaults=dict(source=SRC, sha=NEWSHA, bench=206603))
EngineConfig.objects.filter(name='Luminex').update(source=SRC)
print('Engine:', Engine.objects.get(name='Luminex').sha[:8], '| Tests:', Test.objects.count())
