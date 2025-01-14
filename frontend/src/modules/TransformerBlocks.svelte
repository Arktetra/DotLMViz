<script lang="ts">
    import DottedBlockBase from "../components/DottedBlockBase.svelte";
    import BlockBase from "../components/BlockBase.svelte";

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
</script>

<DottedBlockBase label="Transformer Blocks" style="flex flex-row justify-between items-center w-[30rem] h-[20rem]">
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