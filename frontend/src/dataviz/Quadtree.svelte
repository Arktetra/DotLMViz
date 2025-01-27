<script lang="ts">
    import type { Snippet } from "svelte";
    import type { ScaleLinear } from "d3-scale";
    import type { Quadtree, RandomNumberGenerationSource } from "d3";
    import { quadtree } from "d3";

    type Position = {
        circle: number,
        square: number
    }

    type Margin = {
        top: number,
        left: number,
        bottom: number,
        right: number
    }

    type Props = {
        xScale: ScaleLinear<number, number>,
        yScale: ScaleLinear<number, number>,
        margin: Margin,
        data: ScatterPlotData | HeatMapData,
        searchRadius?: number,
        children: Snippet<[{
            x: { circle: number, square: number },
            y: number | null,
            found: ScatterPlotPoint | HeatMapPoint,
            visible: boolean,
            margin: Margin,
            e: MouseEvent
        }]>
    };

    let {
        xScale,
        yScale,
        margin,
        data,
        searchRadius,
        children
    } : Props = $props();

    let finder: Quadtree<ScatterPlotPoint | HeatMapPoint> | null = $state(null);
    let found: ScatterPlotPoint | HeatMapPoint | null = $state(null);
    let visible: boolean = $state(false);
    let e: MouseEvent = $state({} as MouseEvent);

    $effect(() => {
        if (data) {
            finder = quadtree<ScatterPlotPoint | HeatMapPoint>()
                .x((d) => xScale(d.x))
                .y((d) => yScale(d.y))
                .addAll(data);
        }
    })

    function findItem(evt: MouseEvent) {
        const { layerX, layerY } = evt;

        if (finder) {
            const result = finder.find(layerX, layerY, searchRadius);
            found = result || null;
            visible = result !== null;
            e = evt;
        }
    }

    function getPosition(foundItem: ScatterPlotPoint | HeatMapPoint) {
        if (foundItem?.x) {
            const xPos = xScale(foundItem.x);
            return xPos > 0.9 * xScale.range()[1]
                ? { circle: xPos, square: xPos - 100 }
                : { circle: xPos, square: xPos };
        }

        return null;
    }

    const position: Position | null = $derived(found ? getPosition(found) : null);
    const yPosition: number | null = $derived(found ? yScale(found["y"]) : null);

    $effect(() => {
        $inspect(visible);
    })
</script>

<div
    aria-hidden={true}
    class="bg"
    onmousemove={findItem}
    onblur={() => {
        visible=false;
        found=null;
    }}
    onmouseout={() => {
        visible=false;
        found=null;
    }}>
</div>

{#if found && position && yPosition}
    {@render children({
        x: position,
        y: yPosition,
        found,
        visible,
        margin,
        e
    })}
{/if}

<style>
    .bg {
        position: absolute;
        width: 100%;
        height: 100%;
    }
</style>