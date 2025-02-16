<script lang="ts">
	import { ExpandOutline, MinimizeOutline } from 'flowbite-svelte-icons';
	import { fade } from 'svelte/transition';
	import QuickLink from './QuickLink.svelte';

	let {
		label = 'Untitled',
		href = '/',
		expanded = $bindable<boolean>(),
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();

</script>

<div
	style="border-width: {borderSize};"
	class="relative m-2 rounded-xl border-dashed border-theme transition-colors duration-200"
>
	<div class="absolute end-1 top-[0.2rem] flex flex-row items-center justify-evenly text-theme">
		<QuickLink {href} ostyle="!relative !text-theme" />

		<button onclick={() => (expanded = !expanded)}>
			{#if expanded}
				<MinimizeOutline size={'md'} />
			{:else}
				<ExpandOutline size={'md'} />
			{/if}
		</button>
	</div>
	<span
		class="absolute left-[50%] top-[-1.8rem] translate-x-[-50%] font-main uppercase text-gray-500 {titStyle}"
	>
		{label}
	</span>

	{#if children}
		<div
			transition:fade={{ duration: 500, delay: 500 }}
			class="flex items-center pt-12 transition-all duration-500 {inStyle}"
		>
			{@render children()}
		</div>
	{/if}
</div>
