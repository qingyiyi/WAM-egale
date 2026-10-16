"""Frozen online lifecycle; O1 changes only the installed prefix class."""
from paths import S0, load
from template import TemplateSharedPrefix, clone_function

frozen_online = load('chunk_template_frozen_online', S0 / 'online_shared.py')


class TemplateOnlineShared(frozen_online.OnlineShared):
    install = clone_function(frozen_online.OnlineShared.install,
                             SharedPrefix=TemplateSharedPrefix)


OnlineShared = frozen_online.OnlineShared
