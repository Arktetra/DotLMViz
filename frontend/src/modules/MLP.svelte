<script lang="ts">

    import BlockBase from "../components/BlockBase.svelte";

	let _activeInd = $state(0);

    $effect(() => {
        console.log("Attention head clicked " + _activeInd);
    })
</script>

<h1 class="w-full p-4 mb-5 text-2xl bg-theme-g text-theme font-bold text-center uppercase">Multi-level Perceptron (MLP)</h1>
<BlockBase label="Linear" href="/read/mlp-linear" />
<BlockBase label="new_gelu" href="/read/mlp-new-gelu" />
<BlockBase label="Linear" href="/read/mlp-linear" />
<br />