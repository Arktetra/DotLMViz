<script lang="ts">
    import DottedBlockBase from "../components/DottedBlockBase.svelte";
    import BlockBase from "../components/BlockBase.svelte";
    import ThemeNumberOptions from "../components/ThemeNumberOptions.svelte";

    const _transformerBlock = [
        {
            label: "Attention Head",
            redirect: "/readings/attentionhead"
        },
        {
            label: "MLP",
            redirect: "/readings/mlp"
        }
    ]

    let _activeBlock = $state(0)
</script>

<div class="flex flex-col justify-evenly items-center">
    <span class="text-md text-theme uppercase my-2">Blocks</span>
    <ThemeNumberOptions count={12} bind:activeIndex={_activeBlock} />
    <DottedBlockBase label="Transformer Block {_activeBlock}" style="flex flex-row justify-between items-center w-[30rem] h-[20rem]">
        {#each _transformerBlock as item}
        <div class="flex flex-col justify-evenly items-start h-full">
            <BlockBase label={item.label} height={"10rem"} >
                <h1 class="bg-theme p-4 ">{item.label}</h1>
            </BlockBase>
            <BlockBase label="LN" width={"4rem"} height={"4rem"}>
                <h1 class="bg-theme p-4 ">LayerNorm</h1>
            </BlockBase>
        </div>
        {/each}
    </DottedBlockBase>
</div>