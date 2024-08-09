set -e

PLATFORMS="tods amd intel"
CACHES="l3"
VARY=skew

x/gcp/process.sh skew

for platform in $PLATFORMS; do
    PLAT_DIR=processed/$VARY/$platform
    rename "s/$VARY-$platform-//" $PLAT_DIR/*
    for cache in $CACHES; do
        SRC_PATH=$PLAT_DIR/$cache/2set_vary_${VARY}_${platform}_${cache}
        mv $SRC_PATH/*.csv $PLAT_DIR/$cache
        rm -r $SRC_PATH
    done
done

mv processed/$VARY/tods processed/$VARY/icelake
mv processed/$VARY/amd processed/$VARY/znver4
mv processed/$VARY/intel processed/$VARY/sapphirerapids
