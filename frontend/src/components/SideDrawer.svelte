<script lang="ts">
	import { ChevronDoubleRightOutline, ChevronDoubleLeftOutline } from 'flowbite-svelte-icons';

	const { side = 'right', width = '10rem', children = null } = $props();

	let isOpen: boolean = $state(false);
</script>

<section
	style="width: {width};"
	class={'fixed top-0 z-20 flex h-screen flex-col items-center justify-center border border-l-gray-400 border-r-gray-400 bg-theme-w p-2 transition-transform duration-500' +
		(isOpen
			? ' translate-x-0 '
			: side === 'right'
				? ' translate-x-[95%] '
				: ' translate-x-[-95%] ') +
		(side === 'right' ? ' right-0 pl-6' : ' left-0 pr-6')}
>
	<button
		onclick={() => (isOpen = !isOpen)}
		class={'absolute top-[50%] rounded-md border border-black bg-theme-w ' +
			(side === 'right' ? ' -left-2 ' : ' -right-2 ')}
	>
		{#if side == 'right'}
			<ChevronDoubleRightOutline
				class={'h-8 w-4 text-theme transition-transform duration-500 ' +
					(!isOpen ? ' rotate-180' : ' rotate-0')}
			/>
		{:else}
			<ChevronDoubleLeftOutline
				class={'h-8 w-4 text-theme transition-transform duration-500 ' +
					(!isOpen ? ' rotate-180' : ' rotate-0')}
			/>
		{/if}
	</button>

	{#if children}
		{@render children()}
	{/if}
</section>
