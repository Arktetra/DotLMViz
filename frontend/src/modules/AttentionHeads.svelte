<script lang="ts">
	import { attnHeadCallback } from '../callbacks.svelte';
	import ThemeNumberOptions from '../components/ThemeNumberOptions.svelte';
	import { global_state } from '../state.svelte';

	import { ArrowRightAltOutline } from 'flowbite-svelte-icons';
</script>

<div class="flex h-full w-full flex-col items-center justify-between rounded-lg bg-theme-w p-5">
	<h1
		class="w-full bg-theme-g-alt p-4 text-center font-main-a text-2xl font-bold uppercase text-theme"
	>
		Attention Heads
	</h1>
	<span class="my-5 font-bold text-theme">Active Head : {global_state.active_head}</span>
	<ThemeNumberOptions
		count={12}
		bind:activeIndex={global_state.active_head}
		style="flex flex-row"
		clickEventCb={attnHeadCallback}
	/>
	<div class="flex w-full flex-row items-center justify-evenly">
		{#each { length: 2 } as ind}
			<ArrowRightAltOutline class="h-5 w-5 -rotate-90 animate-pulse text-theme" />
		{/each}
	</div>
</div>
