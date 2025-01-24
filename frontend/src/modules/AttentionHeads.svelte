<script lang="ts">

    import ThemeNumberOptions from "../components/ThemeNumberOptions.svelte";
	import { getAct } from "../routes/fetch.svelte";
	import { activeComponent, global_state } from "../state.svelte";

    /**
     * A function which will fetch the attention pattern and store it
     * in the global state `global_state.data`.
     */
    const getAttnPattern = async () => {
        await getAct("pattern", "attn", global_state.active_block);
        activeComponent.name = "attn";
    }
</script>

<h1 class="w-full p-4 text-2xl bg-theme-g text-theme font-bold text-center uppercase">Attention Heads</h1>
<span class="text-theme font-bold my-5">Active Head : {global_state.active_head}</span>
<ThemeNumberOptions
    count={12}
    bind:activeIndex={global_state.active_head}
    style="grid grid-cols-3 gap-2 p-2"
    clickEventCb={getAttnPattern}
/>
