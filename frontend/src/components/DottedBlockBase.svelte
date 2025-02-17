<script lang="ts">
	import { scale } from 'svelte/transition';

	const {
		label = '',
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();
</script>

<div
	in:scale={{ duration: 300 }}
	style="border-width: {borderSize};"
	class="relative flex h-full items-center justify-center rounded-lg border-dashed border-theme transition-all duration-200 {inStyle}"
>
	<span
		class="absolute left-[50%] top-[-1.5rem] translate-x-[-50%] text-sm font-light uppercase tracking-tighter {titStyle}"
	>
		{label}
	</span>

	{#if children}
		{@render children()}
	{/if}
</div>
