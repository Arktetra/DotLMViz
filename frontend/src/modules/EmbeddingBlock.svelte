<script>
    import DottedBlockBase from "../components/DottedBlockBase.svelte";
    import BlockBase from "../components/BlockBase.svelte";

    const _embeddings = [
        {
        label: "Token Embedding",
        redirect: "/readings/tokenembedding"
        },
        {
        label: "Positional Embedding",
        redirect: "/readings/positionalembedding"
        }
    ]
</script>


<DottedBlockBase label="Embeddings">
    {#each _embeddings as item}
      <BlockBase label={item.label} >
        <h1 class="bg-theme p-4 ">{item.label}</h1>
      </BlockBase>
    {/each}
</DottedBlockBase>