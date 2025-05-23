# Model Registry Migration Guide

## Overview

We've replaced our internal model registry implementation with the dedicated `openai-model-registry` package (v0.4.0+). This provides improved functionality, better error handling, and more accurate model information.

## Breaking Changes

1. **Import Changes**
   - Old: `from openai_structured import ModelRegistry, ModelCapabilities`
   - New: `from openai_model_registry import ModelRegistry, ModelCapabilities`

2. **CLI Command Changes**
   - Old: `openai-structured-refresh`
   - New: `openai-model-registry-update`

3. **Error Types**
   - Error hierarchy and types have changed
   - See the new [error documentation](https://yaniv-golan.github.io/openai-model-registry/api/) for details

4. **Registry Update Method Changes**
   - Old: `registry.check_for_updates()` returned `RegistryUpdateResult`
   - New: `registry.check_for_updates()` returns `RefreshResult` with `RefreshStatus`

5. **Model Capabilities Access**
   - Old: Various direct properties like `context_window`
   - New: Same properties but from different import location

## Migration Steps

1. Install the new package: `pip install openai-model-registry>=0.4.0`
2. Update all imports to use the new library
3. Replace registry CLI usages with the new command
4. Update error handling to use the new error types
