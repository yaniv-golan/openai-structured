# CLI Migration Guide

## Overview

The CLI functionality from `openai-structured` is being moved to a new dedicated package called `ostruct-cli`. This document outlines the migration process and timeline.

## Why Are We Moving the CLI?

We are separating the CLI into its own package to:

- Better separate concerns between the core library and CLI tool
- Allow for independent versioning and updates
- Reduce dependencies for users who only need the library functionality

## Timeline

- **Jan 25, 2025**: Release of `openai-structured` v0.9.2 with deprecation notices DONE
- **Jan 25, 2025**: Release of `ostruct-cli` v1.0.0 with identical functionality DONE
- **Jan 25, 2025**: Release of `openai-structured` v1.0.0 with CLI functionality removed DONE

## Migration Steps

1. Install the new CLI package:

   ```bash
   pip install ostruct-cli
   ```

2. Update any scripts or documentation that reference the old CLI to use the new one.
   The command name and all functionality remain exactly the same:

   ```bash
   # Old (openai-structured <0.9.2):
   ostruct --task template.j2 --file input=data.txt

   # New (ostruct-cli):
   ostruct --task template.j2 --file input=data.txt  # Identical usage
   ```

## Feature Parity

The `ostruct-cli` package provides 100% feature parity with the CLI functionality in `openai-structured`. All commands, options, and behaviors remain identical to ensure a smooth transition.

## Support

- Repository: <https://github.com/yaniv-golan/ostruct>
- Issues: <https://github.com/yaniv-golan/ostruct/issues>
- Documentation: <https://ostruct.readthedocs.io/>

## FAQ

**Q: Do I need to change my CLI usage or scripts?**  
A: No, all commands and options remain exactly the same. Just install the new package.

**Q: Will my existing templates and schemas work?**  
A: Yes, all existing templates, schemas, and configurations will work without modification.

**Q: What happens if I don't migrate?**  
A: The CLI in `openai-structured` will continue to work until v1.0.0, but you should migrate before then to ensure continued functionality.

**Q: Can I install both packages?**  
A: Yes, but it's recommended to migrate to `ostruct-cli` for CLI functionality and use `openai-structured` only for its library features.
