<script lang="ts">
	import { attnHeadCallback } from "../callbacks.svelte";
    import ThemeNumberOptions from "../components/ThemeNumberOptions.svelte";
	import { global_state } from "../state.svelte";

    import { ArrowRightAltOutline } from "flowbite-svelte-icons";
</script>

<h1 class="w-full p-4 text-2xl bg-theme-g text-theme font-bold font-main-a text-center uppercase">Attention Heads</h1>
<span class="text-theme font-bold my-5">Active Head : {global_state.active_head}</span>
<ThemeNumberOptions
    count={12}
    bind:activeIndex={global_state.active_head}
    style="flex flex-row "
    clickEventCb={attnHeadCallback}
/>
<div class="w-full flex flex-row justify-evenly items-center">
    {#each {length: 2} as ind}
        <ArrowRightAltOutline class="h-5 w-5 text-theme animate-pulse" />
    {/each}
</div>

