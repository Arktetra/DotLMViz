<script lang="ts">

    import ThemeNumberOptions from "../components/ThemeNumberOptions.svelte";

	let _activeInd = $state(0);

    $effect(() => {
        console.log("Attention head clicked " + _activeInd);
    })
</script>

<h1 class="w-full p-4 text-2xl bg-theme-g text-theme font-bold text-center uppercase">Attention Heads</h1>
<span class="text-theme font-bold my-5">Active Head : {_activeInd}</span>
<ThemeNumberOptions count={12} bind:activeIndex={_activeInd} style="grid grid-cols-3 gap-2 p-2" />
