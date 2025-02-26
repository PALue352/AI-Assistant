import logging

# Version code for tracking
VERSION = "v1.001"  # Initial version for adding version tracking

logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(name)s - %(levelname)s - [Version {VERSION}] - %(message)s')
logger = logging.getLogger(__name__)

class PluginManager:
    def __init__(self):
        logger.info(f"PluginManager initializing... [Version {VERSION}]")
        self.plugins = {}
        self.overseer = None
        logger.info(f"PluginManager initialized. [Version {VERSION}]")

    def set_overseer(self, overseer):
        self.overseer = overseer
        logger.info(f"PluginManager linked to Overseer. [Version {VERSION}]")

    def load_plugin(self, plugin_name, plugin_class):
        try:
            self.plugins[plugin_name] = plugin_class(self.overseer)
            logger.info(f"Loaded plugin {plugin_name}. [Version {VERSION}]")
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}. [Version {VERSION}]")

    def unload_plugin(self, plugin_name):
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"Unloaded plugin {plugin_name}. [Version {VERSION}]")
        else:
            logger.warning(f"Plugin {plugin_name} not found. [Version {VERSION}]")

if __name__ == "__main__":
    manager = PluginManager()
    # Example plugin class (simplified)
    class TestPlugin:
        def __init__(self, overseer):
            pass
    manager.load_plugin("test", TestPlugin)
    manager.unload_plugin("test")