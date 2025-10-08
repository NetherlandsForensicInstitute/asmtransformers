# ATF example

This demonstrates how `asmtransformers` can find symbols across compiler optimization settings. We compiled the [`atf`](https://github.com/ARM-software/arm-trusted-firmware) with different compiler optimization levels and loaded them in Ghidra. We generated stripped and non-stripped binaries. We then loaded all the symbols of `bl31` when compiled with `-O2` into `sententia-demo-O2-bl31.db` using the provided `AddSymbolsToSententiaDB.java` script. This database will be used to compare against other optimization levels to see if we can still identify the correct functions.

To use this example, first set up the [**citatio**](./../../citatio) server and [**sententia**](./../../sententia) Ghidra plugin as described in their respective READMEs.

Now we start the **citatio** server with  `bl31` `-O2` database.

```bash
cd citatio
CITATIO_SQLITE_DATABASE=../examples/atf/sententia-demo-O2-bl31.db pdm run fastapi run citatio
```

Import the archived Ghidra project `Sententia_arm-trusted-firmware_demo.gar` in Ghidra (from an empty project window select **Restore project**).

Open `arm-trusted-firmware/O2/bl31.elf`:

- Two very similar functions, for example `nor_unlock` and `nor_erase`, get very similar scores.
- `plat_arm_get_mmap` and `get_arm_std_svc_args` are too short and too similar to distinguish. In general, short functions are hard to distinguish from each other.

Open `arm-trusted-firmware/Os/bl31.elf`:

- `nor_unlock` and `nor_erase` still get correctly identified, but it becomes a lot harder.

Open `arm-trusted-firmware/O3/bl31.elf`:

- Something like `smmuv3_security_init` and `memcpy` is easy between O2 and Os, but a lot harder in O3 (but still is the top candidate!)