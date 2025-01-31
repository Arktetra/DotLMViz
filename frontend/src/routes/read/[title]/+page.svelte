<script lang="ts">
	import { page } from '$app/state';
	import { content } from '$lib/content';

	console.log(page.params);
	const title = page.params.title.split('-').join(' ');
	const description = content[page.params.title] ?? 'NO information available for this topic!!';
</script>

<div class="flex flex-col items-center justify-evenly pt-10 font-main-a">
	<span class="text-md uppercase">Readings</span>
	<div
		class="my-10 min-h-[80vh] w-[80%] rounded-sm border-2 border-theme-g p-10 text-justify text-lg tracking-wider shadow-lg"
	>
		<h1 class="mb-10 text-center font-main text-4xl font-extrabold uppercase text-theme underline">
			{title}
		</h1>
		<p class="h-full w-full">
			{@html description}
		</p>
	</div>
</div>
