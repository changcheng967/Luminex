import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'OpenSite.settings')
django.setup()
from OpenBench.models import Engine, EngineConfig, Test, Result

NEWSHA = 'a64ddacd0b13a5adc64dbf2d39c617ee9091364d'
SRC = f'https://github.com/changcheng967/Luminex/archive/{NEWSHA}.zip'
Engine.objects.update_or_create(name='Luminex', defaults=dict(source=SRC, sha=NEWSHA, bench=206603))
EngineConfig.objects.filter(name='Luminex').update(source=SRC)
print('Engine:', Engine.objects.get(name='Luminex').sha[:8])
print('Tests:', Test.objects.count())
