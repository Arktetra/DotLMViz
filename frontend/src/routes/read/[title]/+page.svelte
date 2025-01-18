<script lang="ts">
	import { page } from '$app/state';
	console.log(page.params)
	const title = (page.params.title.split("-")).join(" ");
</script>

<div class="flex flex-col justify-evenly items-center">
	<span class="text-md uppercase">Readings</span>
	<div class="w-[80%] min-h-[80vh] p-10 my-10 text-justify text-lg tracking-wider border-2 border-theme-g shadow-lg rounded-sm">
		<h1 class="font-extrabold mb-10 text-center text-4xl text-theme uppercase">{title}</h1>
		<p class="w-full h-full">
			This is the page containing the information about <b>{title}</b>. Let's
			consider <b>{title}</b> is used in this page to do this... and that...
		</p>
	</div>
</div>

